#include <limits>

#include "implementation.h"
#include "util_gpu_err_check.cuh"

#include "stdio.h"

#ifndef restrict
#define restrict __restrict__
#endif

constexpr unsigned MASK_ALL = 0xffff'ffffu;

// `warpSize` is not actually a compile-time constant, so some of the arithmetic
// is opaque to the compiler; make it constexpr to get better optimisation.
constexpr unsigned WARP_SIZE = 32;

constexpr unsigned CHUNK_SIZE = 512;
constexpr unsigned SUBCHUNK_SIZE = WARP_SIZE;

enum class ChunkState : int32_t
{
	None = 0,
	Local,
	Finished,
};

struct Status_t
{
	int32_t sum;
	ChunkState state;
};

static_assert(sizeof(Status_t) == sizeof(uint64_t));

static __device__ __forceinline__ Status_t load_status(volatile const uint64_t* ptr)
{
	Status_t ret;
	*((uint64_t*) &ret) = __ldcv((const uint64_t*) ptr);
	return ret;
}

static __device__ __forceinline__ void store_status(volatile uint64_t* ptr, Status_t val)
{
	__stcg((uint64_t*) ptr, *((const uint64_t*) &val));
}


// this is a per-warp thing
template <unsigned Lanes = WARP_SIZE, typename T>
static __forceinline__ __device__ T warp_prefix_sum(T value)
{
	// use shuffles to go faster
	const uint8_t lane_idx = threadIdx.x & (WARP_SIZE - 1);
	for(uint8_t ofs = 1; ofs < Lanes; ofs *= 2)
	{
		const auto tmp = __shfl_up_sync(MASK_ALL, value, ofs);
		if(lane_idx >= ofs)
			value += tmp;
	}
	return value;
}


static __device__ int32_t sum_chunk(int32_t thread_value, unsigned num_elems)
{
	// one extra guy at the front so that we don't need to branch on whether
	// we are the first subchunk or not later
	__shared__ int32_t partials[1 + SUBCHUNK_SIZE];

	if(threadIdx.x < SUBCHUNK_SIZE)
		partials[1 + threadIdx.x] = 0;

	if(threadIdx.x == 0)
		partials[0] = 0;

	const uint8_t warp_idx = threadIdx.x / WARP_SIZE;
	const uint8_t lane_idx = threadIdx.x & (WARP_SIZE - 1);

	const auto result = warp_prefix_sum(thread_value);

	__syncthreads();

	// for the last guy in each subchunk, write the result to the temp array
	if(lane_idx == SUBCHUNK_SIZE - 1 && threadIdx.x < num_elems)
		partials[1 + warp_idx] = result;

	// make sure the results for all warps are stored in the array
	__syncthreads();

	// note: since we only have 512/32 = 16 partials, just do half a warp to save some instructions.
	if(warp_idx == 0)
		partials[1 + lane_idx] = warp_prefix_sum<(CHUNK_SIZE/WARP_SIZE)>(partials[1 + lane_idx]);

	__syncthreads();

	// now each warp adds the prefix-summed subresult to its own subchunk of 32 values
	// because of the +1 we did above, partials[0] contains 0, so it's a no-op for the first warp.
	return result + partials[warp_idx];
}



static __device__ void chunk_lookbehind_parallel(
	uint32_t block_idx,
	volatile uint64_t* restrict statuses,
	volatile int32_t* block_exclusive_sum
)
{
	__shared__ volatile uint32_t offset;

	const uint8_t warp_idx = threadIdx.x / WARP_SIZE;
	const uint8_t lane_idx = threadIdx.x & (WARP_SIZE - 1);

	// if this is the first block, there's nothing to do.
	if(block_idx == 0)
		return;

	if(threadIdx.x == 0)
		offset = 0;

	__syncthreads();

	// look behind up to 32 chunks at a time (one warp)
	if(warp_idx != 0)
		return;

	while(true)
	{
		// if the current lane does not have a predecessor, then it should not affect
		// the outcome -- ie. it should not be NONE.
		const auto pred_status = [&]() -> Status_t {
			if(block_idx > lane_idx + offset)
			{
				// note that this indexing scheme makes lane 0 control a higher pred_idx than lane 31.
				// for example, if we are at block 69, then lane0 reads 68, lane1 reads 67, etc.
				return load_status(&statuses[block_idx - offset - lane_idx - 1]);
			}
			else
			{
				return { 0, ChunkState::Local };
			}
		}();

		// wait for all warps to do the thing. __all_sync should synchronise (otherwise it's a shit name)
		const auto all_preds_aggregated = __all_sync(MASK_ALL, (pred_status.state != ChunkState::None));

		// if not all preds were aggregated, we need to keep waiting
		if(not all_preds_aggregated)
			continue;

		const auto finished_lanes = __ballot_sync(MASK_ALL, (pred_status.state == ChunkState::Finished));
		const auto last_finished_pred_chunk = __ffs(finished_lanes);

		if(finished_lanes)
		{
			// first, sum all threads in the warp, then take the lane that contains the correct value.
			const auto pred_sum = __reduce_add_sync(MASK_ALL,
				lane_idx < last_finished_pred_chunk
					? pred_status.sum
					: 0
			);

			if(lane_idx + 1 == last_finished_pred_chunk)
				*block_exclusive_sum += pred_sum;

			// we're done, gtfo
			break;
		}
		else
		{
			// add the partial sums of the predecessors to our exclusive sum
			// (of course, but only the last thread in the group)
			const auto pred_sum = __reduce_add_sync(MASK_ALL, pred_status.sum);
			if(lane_idx + 1 == min(WARP_SIZE, block_idx - offset))
				*block_exclusive_sum += pred_sum;

			if(lane_idx == 0)
				offset += WARP_SIZE;

			// check whether we should stop -- this is when there's no more work
			// for *any* thread in the warp to do.
			if(__shfl_sync(MASK_ALL, offset, 0) >= block_idx)
				break;
		}
	}
}



__global__
__launch_bounds__(CHUNK_SIZE)
void kernel(
	const int32_t* restrict in,
	int32_t* restrict out,
	uint64_t* restrict statuses,
	uint32_t* restrict block_exec_counter,
	uint32_t num_elems)
{
	// only the first thread in the block should get the unique block id
	__shared__ uint32_t _block_idx;
	__shared__ int32_t block_exclusive_sum;

	if(threadIdx.x == 0)
	{
		// set our own state to invalid
		const auto asdf = atomicAdd(block_exec_counter, 1);
		store_status(&statuses[asdf], { 0, ChunkState::None });
		_block_idx = asdf;

		block_exclusive_sum = 0;
	}

	__syncthreads();

	// the compiler is apparently not smart enough to cache this (even though it's not volatile),
	// so do the caching manually so we stop hitting shared memory
	const auto block_idx = _block_idx;
	const auto thread_idx = threadIdx.x;

	const uint32_t chunk_ofs = block_idx * CHUNK_SIZE;
	const uint32_t chunk_size = min(CHUNK_SIZE, num_elems - chunk_ofs);

	const bool is_last = (thread_idx == chunk_size - 1);

	// perform chunkwise (per block) scan
	const auto input_idx = chunk_ofs + min(thread_idx, chunk_size - 1);
	const auto local_sum = sum_chunk(__ldcv(&in[input_idx]), chunk_size);

	// update the local sum for this block with the last guy
	if(is_last)
		store_status(&statuses[block_idx], { local_sum, ChunkState::Local });

	// do the lookbehind to compute the *exclusive* sum for this block,
	// and add it to the local sum (for this thread)
	chunk_lookbehind_parallel(block_idx, statuses, &block_exclusive_sum);
	__syncthreads();

	const auto final_sum = local_sum + block_exclusive_sum;

	// now we have the final result, update our status to "finished"
	if(is_last)
		store_status(&statuses[block_idx], { final_sum, ChunkState::Finished });

	// now every thread just writes its value to global memory.
	if(chunk_size == CHUNK_SIZE)
		__stcg(&out[input_idx], final_sum);
	else if(const auto thread_ofs = chunk_ofs + thread_idx; thread_ofs < num_elems)
		__stcg(&out[thread_ofs], final_sum);
}



/*
 * @param d_input: input array on device
 * @param d_output: output array on device
 * @param size: number of elements in the input array
 */
void implementation(const int32_t* d_input, int32_t* d_output, size_t size)
{
	if(size > std::numeric_limits<uint32_t>::max())
		printf("oh no, too many elements ):\n");

	const auto num_chunks = (size + CHUNK_SIZE - 1) / CHUNK_SIZE;

	// one malloc for both thingies
	uint64_t* status_array = nullptr;
	gpu_err_check(cudaMalloc((void**) &status_array, num_chunks * sizeof(uint64_t) + sizeof(uint32_t)));
	gpu_err_check(cudaMemset(status_array, 0, num_chunks * sizeof(uint64_t) + sizeof(uint32_t)));

	uint32_t* block_exec_counter = (uint32_t*) &status_array[num_chunks];

	kernel<<<num_chunks, CHUNK_SIZE>>>(d_input, d_output, status_array, block_exec_counter, size);

	gpu_err_check(cudaDeviceSynchronize());
	gpu_err_check(cudaFree(status_array));
}
