#include <cinttypes>

#include "common.hpp"

/// @brief  Every N elements should be padded by 1.
///
/// @detailed   For every valid N (i.e. greater than or equal to 1), insert
///             padding between every N elements to prevent bank conflicts. Note
///             that the performance drops a bit for adding padding of {16, 32, 64}
///             and drops a lot for non-power-of-2 padding (e.g. 15).
#define PADDED_EVERY_N(x, n)            ((n) <= 0) ? (x) : (((n) + 1) * ((x) / (n)) + (x) % (n))
#define PADDED(x)                       PADDED_EVERY_N((x), 16)

/// @brief  Ceiling divide size_t.
///
/// Assuming numerator + denominator does not overflow.
/// Source: https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
#define CEIL_DIV(n, d)                  (((n) + (d) - 1) / (d))

__device__ __forceinline__ int32_t
warp_scan(int32_t my_val)
{
    int t_lane = threadIdx.x % 32;
    for (size_t offset = 1; offset < 32; offset *= 2) {
        const int32_t prev_val = __shfl_up_sync(0xFFFFFFFF, my_val, offset);
        if (t_lane >= offset) {
            my_val += prev_val;
        }
    }
    int32_t prev_val = __shfl_up_sync(0xFFFFFFFF, my_val, 1);
    return t_lane == 0 ? 0 : prev_val;
}

/// @brief Algorithm to scan a block.
///
/// @detailed   See Figure 2(a) in the Decoupled Look-back paper
///
/// @param  d_output: int32_t*
///             array of outputs. Potentially, we could allow this to be nullable
///             to prevent copying d_input to d_output if the latter already has
///             the data (e.g. if we are using reduction_array)
/// @param  reduction_array: int32_t*
///             array of the maximum of each thread block. NULL if we do not
///             want to collect this information.
///
/// @note   We could improve the efficiency of this because this is really four
///         functions smooshed together.
/// 
///         1. Scan 1024 elements from d_input into d_output, record max in reduction_array
///         2. Scan up-to 1024 elements from d_input into d_output
///         3. Scan up-to 1024 elements from reduction_array into reduction_array
///         4. (optional) the above but with 1024 fixed elements. Only if we had
///            more than 1024x1024 elements.
template<size_t max_num_threads>
__global__ void
block_scan(int32_t const *d_input, int32_t *d_output,
                    size_t const size, int32_t *const reduction_array)
{
    // Shared memory is supposed to be 100x faster than global memory
    constexpr const int num_warps = CEIL_DIV(max_num_threads, 32);
    __shared__ int32_t buffer_max_of_warps[PADDED(num_warps)];
    const int global_idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int t_id = threadIdx.x;
    const int t_lane = t_id % 32;
    const int warp_id = t_id / 32;

    // Ignore indices that are out of bounds on the main array
    int32_t r0 = 0;
    int32_t r1 = 0;
    int32_t r2 = 0;
    int32_t r3 = 0;
    if (global_idx + 3 < size) {
        // Copy to warp and reduce warp
        r0 =      d_input[global_idx + 0];
        r1 = r0 + d_input[global_idx + 1];
        r2 = r1 + d_input[global_idx + 2];
        r3 = r2 + d_input[global_idx + 3];
    } else if (global_idx + 2 < size) {
        r0 =      d_input[global_idx + 0];
        r1 = r0 + d_input[global_idx + 1];
        r2 = r1 + d_input[global_idx + 2];
        r3 = r2;
    } else if (global_idx + 1 < size) {
        r0 =      d_input[global_idx + 0];
        r1 = r0 + d_input[global_idx + 1];
        r2 = r1;
        r3 = r1;
    } else if (global_idx + 0 < size) {
        r0 =      d_input[global_idx + 0];
        r1 = r0;
        r2 = r0;
        r3 = r0;
    } else {
        return;
    }
    __syncwarp();

    // Warp-reduce
    int32_t max_of_prev_thread = warp_scan(r3);
    r0 += max_of_prev_thread;
    r1 += max_of_prev_thread;
    r2 += max_of_prev_thread;
    r3 += max_of_prev_thread;
    if (t_lane == 31) {
        buffer_max_of_warps[PADDED(warp_id)] = r3;
    }
    __syncthreads();

    // Strided Propagations
    if (warp_id == 0) {
        // 1. Read from max of each warp
        int32_t my_val = t_id < num_warps ? buffer_max_of_warps[PADDED(t_id)] : 0;
        __syncwarp();
        // 2. Reduce max of each warp
        for (size_t offset = 1; offset < num_warps; offset *= 2) {
            const int32_t other_val = __shfl_up_sync(0xFFFFFFFF, my_val, offset);
            if (t_lane >= offset) {
                my_val += other_val;
            }
        }
        // 3. Write max of each warp
        if (t_id < num_warps) {
            buffer_max_of_warps[PADDED(t_id)] = my_val;
        }
    }
    __syncthreads();

    // Propagation Fan to all non-first warps
    // If not the first warp and not the last element of the warp
    if (warp_id != 0) {
        int32_t max_of_prev_warp = buffer_max_of_warps[PADDED(warp_id - 1)];
        r0 += max_of_prev_warp;
        r1 += max_of_prev_warp;
        r2 += max_of_prev_warp;
        r3 += max_of_prev_warp;
    }
    // If not the first thread, 
    __syncthreads();

    // Last thread writes its last element to the global array (i.e. total reduction)
    if (t_id == max_num_threads - 1 && reduction_array != NULL) {
        reduction_array[blockIdx.x] = r3;
    }
    d_output[global_idx    ] = r0;
    d_output[global_idx + 1] = r1;
    d_output[global_idx + 2] = r2;
    d_output[global_idx + 3] = r3;
}

/// @brief  Add the last element of the previous block to the current block.
__global__ void
add_to_all_upto1024(int32_t *const d_output, size_t const size, int32_t const *const reduction_array)
{
    const int dst_idx = 4 * ((blockIdx.x + 1) * blockDim.x + threadIdx.x);
    const int src_idx = blockIdx.x;

    // Add last element of prev block to current
    if (dst_idx + 3 < size) {
        d_output[dst_idx] += reduction_array[src_idx];
        d_output[dst_idx + 1] += reduction_array[src_idx];
        d_output[dst_idx + 2] += reduction_array[src_idx];
        d_output[dst_idx + 3] += reduction_array[src_idx];
    } else {
        if (dst_idx + 2 < size) {
            d_output[dst_idx + 2] += reduction_array[src_idx];
        }
        if (dst_idx + 1 < size) {
            d_output[dst_idx + 1] += reduction_array[src_idx];
        }
        if (dst_idx < size) {
            d_output[dst_idx] += reduction_array[src_idx];
        }
    }
}

void
hierarchical_scan(const int32_t *d_input, int32_t *d_output, size_t size)
{
    constexpr size_t MAX_THREADS = 256;

    size_t num_blocks = CEIL_DIV(size, 4 * MAX_THREADS);
    if (size > 4 * MAX_THREADS) {
        int32_t *reductions = NULL;
        /*assert no error*/cudaMalloc((void **)&reductions, num_blocks * sizeof(int32_t));
        block_scan<MAX_THREADS><<<num_blocks, MAX_THREADS>>>(d_input, d_output, size, reductions);
        cudaDeviceSynchronize();
        // Perform scan on reductions
        hierarchical_scan(reductions, reductions, num_blocks);
        // Add scan to blocks
        add_to_all_upto1024<<<num_blocks - 1, MAX_THREADS>>>(d_output, size, reductions);
        cudaDeviceSynchronize();
        /*assert no error*/cudaFree(reductions);
    } else {
        block_scan<MAX_THREADS><<<1, size>>>(d_input, d_output, size, NULL);
        cudaDeviceSynchronize();
    }
}

void
impl_baseline(const int32_t *d_input, int32_t *d_output, size_t size)
{
    if (size > std::numeric_limits<uint32_t>::max())
        printf("oh no, too many elements ):\n");

    hierarchical_scan(d_input, d_output, size);

    cuda_check(cudaDeviceSynchronize());
}
