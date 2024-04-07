// adapted from nvidia's CUB implementation

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda/std/functional>
#include <type_traits>
#include <vector_types.h>

static constexpr auto WARP_THREADS = 32;
static constexpr auto LOG_SMEM_BANKS = 5;
static constexpr auto ITEMS_PER_THREAD = 15;
static constexpr auto THREADS_PER_BLOCK = 128;
using Sum = ::cuda::std::plus<>;

template <typename T,
          typename U,
          typename V = typename std::common_type_t<T, U>>
constexpr V
mixedMin(
    const T &a,
    const U &b,
    typename std::enable_if_t<
        !std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>> * = nullptr)
{
    return std::min<V>(a, b);
}

template <typename T,
          typename U,
          typename V = typename std::common_type_t<T, U>>
constexpr V
mixedMax(
    const T &a,
    const U &b,
    typename std::enable_if_t<
        !std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>> * = nullptr)
{
    return std::max<V>(a, b);
}

__host__ __device__ __forceinline__ constexpr int
intlog2(int n)
{

    int ret = 0;
    while (n > 1) {
        n >>= 1;
        ++ret;
    }
    return ret;
}

__host__ __device__ constexpr bool
powerOfTwo(int n)
{
    return (n & (n - 1)) == 0;
}

enum Category {
    NOT_A_NUMBER,
    SIGNED_INTEGER,
    UNSIGNED_INTEGER,
};
template <Category _CATEGORY,
          bool _PRIMITIVE,
          typename _UnsignedBits,
          typename T>
struct BaseTraits {

    static const Category CATEGORY = _CATEGORY;
    enum {
        PRIMITIVE = _PRIMITIVE,
    };
};
enum ScanTileStatus {
    SCAN_TILE_OOB,
    SCAN_TILE_INVALID,
    SCAN_TILE_PARTIAL,
    SCAN_TILE_INCLUSIVE,
};
template <typename T>
struct NumericTraits : BaseTraits<NOT_A_NUMBER, false, T, T> {};
template <>
struct NumericTraits<int>
    : BaseTraits<SIGNED_INTEGER, true, unsigned int, int> {};
template <>
struct NumericTraits<unsigned int>
    : BaseTraits<UNSIGNED_INTEGER, true, unsigned int, unsigned int> {};
template <typename T>
struct Traits : NumericTraits<typename std::remove_cv<T>::type> {};

template <int Delay, unsigned int GridThreshold = 500>
__device__ __forceinline__ void
delay()
{

    if (Delay > 0) {
        if (gridDim.x < GridThreshold) {
            __threadfence_block();
        } else {
            __nanosleep(Delay);
        }
    }
}
static __device__ __forceinline__ void
store_relaxed(uint2 *ptr, uint2 val)
{
    asm volatile("st.relaxed.gpu.v2.u32 [%0], {%1, %2};"
                 :
                 : "l"(ptr), "r"(val.x), "r"(val.y)
                 : "memory");
}
static __device__ __forceinline__ uint2
load_relaxed(uint2 const *ptr)
{
    uint2 retval;
    asm volatile("ld.relaxed.gpu.v2.u32 {%0, %1}, [%2];"
                 : "=r"(retval.x), "=r"(retval.y)
                 : "l"(ptr)
                 : "memory");
    return retval;
}
template <typename T,
          typename U,
          typename ::cuda::std::enable_if<
              ::cuda::std::is_trivially_copyable<T>::value,
              int>::type = 0>
__host__ __device__ void
uninitialized_copy(T *ptr, U &&val)
{
    *ptr = ::cuda::std::forward<U>(val);
}

__device__ __forceinline__ unsigned int
LaneId()
{
    unsigned int ret;
    asm("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

struct ScanTileState {
    using StatusWord = unsigned int;
    using TxnWord = uint2;
    static constexpr int TILE_STATUS_PADDING = 32;

    struct TileDescriptor {
        StatusWord status;
        int32_t value;
    };

    __host__ __device__ __forceinline__ static size_t
    AllocationSize(size_t num_tiles)
    {
        auto allocation_size =
            static_cast<size_t>(num_tiles + TILE_STATUS_PADDING) *
            sizeof(TxnWord);
        return allocation_size;
    }

    TxnWord *d_tile_descriptors;

    __host__ __device__ __forceinline__
    ScanTileState()
        : d_tile_descriptors(nullptr)
    {
    }

    __host__ __device__ __forceinline__ cudaError_t
    Init(int /*num_tiles*/, void *d_temp_storage, size_t /*temp_storage_bytes*/)
    {
        d_tile_descriptors = reinterpret_cast<TxnWord *>(d_temp_storage);
        return cudaSuccess;
    }

    __device__ __forceinline__ void
    InitializeStatus(int num_tiles)
    {
        auto tile_idx =
            static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);

        auto val = TxnWord();
        TileDescriptor *descriptor = reinterpret_cast<TileDescriptor *>(&val);

        if (tile_idx < num_tiles) {
            // Not-yet-set
            descriptor->status = StatusWord(SCAN_TILE_INVALID);
            d_tile_descriptors[TILE_STATUS_PADDING + tile_idx] = val;
        }

        if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING)) {
            // Padding
            descriptor->status = StatusWord(SCAN_TILE_OOB);
            d_tile_descriptors[threadIdx.x] = val;
        }
    }

    __device__ __forceinline__ void
    SetInclusive(int tile_idx, int32_t tile_inclusive)
    {
        TileDescriptor tile_descriptor{SCAN_TILE_INCLUSIVE, tile_inclusive};

        TxnWord alias;
        *reinterpret_cast<TileDescriptor *>(&alias) = tile_descriptor;

        store_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx,
                      alias);
    }

    __device__ __forceinline__ void
    SetPartial(int tile_idx, int32_t tile_partial)
    {
        TileDescriptor tile_descriptor{SCAN_TILE_PARTIAL, tile_partial};

        TxnWord alias;
        *reinterpret_cast<TileDescriptor *>(&alias) = tile_descriptor;

        store_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx,
                      alias);
    }

    __device__ __forceinline__ void
    WaitForValid(int tile_idx, StatusWord &status, int32_t &value)
    {
        TileDescriptor tile_descriptor;

        {
            TxnWord alias = load_relaxed(d_tile_descriptors +
                                         TILE_STATUS_PADDING + tile_idx);
            tile_descriptor = reinterpret_cast<TileDescriptor &>(alias);
        }

        while (__any_sync(0xffffffff,
                          (tile_descriptor.status == SCAN_TILE_INVALID))) {
            delay<350>();
            TxnWord alias = load_relaxed(d_tile_descriptors +
                                         TILE_STATUS_PADDING + tile_idx);
            tile_descriptor = reinterpret_cast<TileDescriptor &>(alias);
        }

        status = tile_descriptor.status;
        value = tile_descriptor.value;
    }
    __device__ __forceinline__ int32_t
    LoadValid(int tile_idx)
    {
        TxnWord alias = d_tile_descriptors[TILE_STATUS_PADDING + tile_idx];
        TileDescriptor tile_descriptor =
            reinterpret_cast<TileDescriptor &>(alias);
        return tile_descriptor.value;
    }
};

struct ActualParam {
    static constexpr auto THREAD_ITEMS = std::max(
        1,
        std::min(ITEMS_PER_THREAD * 4 / static_cast<int>(sizeof(int32_t)),
                 ITEMS_PER_THREAD * 2));
    static constexpr auto BLOCK_THREADS = mixedMin(
        THREADS_PER_BLOCK,
        (((1024 * 48) / (sizeof(int32_t) * THREAD_ITEMS)) + 31) / 32 * 32);
};

#if defined(__CUDACC__)
inline cudaError_t
gpuAssert(cudaError_t code, const char *filename, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,
                "cuda assertion failed (%s: %d): %s\n",
                filename,
                line,
                cudaGetErrorString(code));
        fflush(stderr);
        if (abort) {
            std::exit(code);
        }
    }
    return code;
}
#define check_cuda_error(code) gpuAssert((code), __FILE__, __LINE__);
#endif

template <int A> struct Int2Type {
    static constexpr auto VALUE = A;
};

template <typename T> struct AlignBytes {
    struct Pad {
        T val;
        char byte;
    };

    static constexpr auto ALIGN_BYTES = sizeof(Pad) - sizeof(T);
    using Type = T;
};

template <typename T> struct UnitWord {
    static constexpr auto ALIGN_BYTES = AlignBytes<T>::ALIGN_BYTES;

    template <typename Unit> struct IsMultiple {
        static constexpr auto UNIT_ALIGN_BYTES = AlignBytes<Unit>::ALIGN_BYTES;
        static constexpr auto IS_MULTIPLE =
            (sizeof(T) % sizeof(Unit) == 0) &&
            (int(ALIGN_BYTES) % int(UNIT_ALIGN_BYTES) == 0);
    };
    using ShuffleWord =
        std::conditional_t<IsMultiple<int>::IS_MULTIPLE,
                           unsigned int,
                           std::conditional_t<IsMultiple<short>::IS_MULTIPLE,
                                              unsigned short,
                                              unsigned char>>;
    using VolatileWord = std::conditional_t<IsMultiple<long long>::IS_MULTIPLE,
                                            unsigned long long,
                                            ShuffleWord>;
    using DeviceWord = std::conditional_t<IsMultiple<longlong2>::IS_MULTIPLE,
                                          ulonglong2,
                                          VolatileWord>;
};

template <typename T> struct Uninitialized {
    using DeviceWord = typename UnitWord<T>::DeviceWord;

    static constexpr auto DATA_SIZE = sizeof(T);
    static constexpr auto WORD_SIZE = sizeof(DeviceWord);
    static constexpr auto WORDS = DATA_SIZE / WORD_SIZE;
    DeviceWord storage[WORDS];

    __host__ __device__ __forceinline__ T &
    Alias()
    {
        return reinterpret_cast<T &>(*this);
    }
};

class BlockExchange {
private:
    static constexpr auto WARPS =
        (ActualParam::BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS;
    static constexpr auto TIME_SLICED_ITEMS =
        ActualParam::BLOCK_THREADS * ActualParam::THREAD_ITEMS;

    static constexpr auto WARP_TIME_SLICED_THREADS =
        mixedMin(ActualParam::BLOCK_THREADS, WARP_THREADS);
    static constexpr auto WARP_TIME_SLICED_ITEMS =
        WARP_TIME_SLICED_THREADS * ActualParam::THREAD_ITEMS;

    static constexpr auto INSERT_PADDING =
        (ActualParam::THREAD_ITEMS > 4) &&
        powerOfTwo(ActualParam::THREAD_ITEMS);
    static constexpr auto PADDING_ITEMS =
        (INSERT_PADDING) ? (TIME_SLICED_ITEMS >> LOG_SMEM_BANKS) : 0;

    struct __align__(16) _TempStorage
    {
        int32_t buff[TIME_SLICED_ITEMS + PADDING_ITEMS];
    };

private:
    _TempStorage &temp_storage;

    unsigned int thread_id;
    unsigned int lane_id;
    unsigned int warp_id;
    unsigned int warp_offset;

public:
    using TempStorage = Uninitialized<_TempStorage>;
    __device__ __forceinline__
    BlockExchange(TempStorage &temp_storage)
        : temp_storage(temp_storage.Alias()),
          thread_id(threadIdx.x),
          lane_id(LaneId()),
          warp_id((WARPS == 1) ? 0 : thread_id / WARP_THREADS),
          warp_offset(warp_id * WARP_TIME_SLICED_ITEMS)
    {
    }

    __device__ __forceinline__ void
    StripedToBlocked(int32_t (&input_items)[ActualParam::THREAD_ITEMS],
                     int32_t (&output_items)[ActualParam::THREAD_ITEMS])
    {
#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            auto item_offset = i * ActualParam::BLOCK_THREADS + thread_id;
            if (INSERT_PADDING)
                item_offset += item_offset >> LOG_SMEM_BANKS;
            uninitialized_copy(temp_storage.buff + item_offset, input_items[i]);
        }

        __syncthreads();

#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            auto item_offset = (thread_id * ActualParam::THREAD_ITEMS) + i;
            if (INSERT_PADDING)
                item_offset += item_offset >> LOG_SMEM_BANKS;
            output_items[i] = temp_storage.buff[item_offset];
        }
    }

    __device__ __forceinline__ void
    BlockedToStriped(int32_t (&items)[ActualParam::THREAD_ITEMS])
    {
#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            auto item_offset = (thread_id * ActualParam::THREAD_ITEMS) + i;
            if (INSERT_PADDING)
                item_offset += item_offset >> LOG_SMEM_BANKS;
            uninitialized_copy(temp_storage.buff + item_offset, items[i]);
        }

        __syncthreads();

#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            auto item_offset = i * ActualParam::BLOCK_THREADS + thread_id;
            if (INSERT_PADDING)
                item_offset += item_offset >> LOG_SMEM_BANKS;
            items[i] = temp_storage.buff[item_offset];
        }
    }

    __device__ __forceinline__ void
    BlockedToWarpStriped(int32_t (&items)[ActualParam::THREAD_ITEMS])
    {
#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            auto item_offset =
                warp_offset + i + (lane_id * ActualParam::THREAD_ITEMS);
            if (INSERT_PADDING)
                item_offset += item_offset >> LOG_SMEM_BANKS;
            uninitialized_copy(temp_storage.buff + item_offset, items[i]);
        }

        __syncwarp(0xffffffff);

#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            auto item_offset =
                warp_offset + (i * WARP_TIME_SLICED_THREADS) + lane_id;
            if (INSERT_PADDING)
                item_offset += item_offset >> LOG_SMEM_BANKS;
            items[i] = temp_storage.buff[item_offset];
        }
    }
};

class BlockLoad {
private:
    using _TempStorage = typename BlockExchange::TempStorage;
    _TempStorage &temp_storage;
    unsigned long thread_id;

public:
    using TempStorage = Uninitialized<_TempStorage>;

    __device__ __forceinline__
    BlockLoad(TempStorage &temp_storage)
        : temp_storage(temp_storage.Alias()),
          thread_id(threadIdx.x)
    {
    }

    __device__ __forceinline__ void
    Load(const int32_t *block_itr, int32_t (&items)[ActualParam::THREAD_ITEMS])
    {
#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            items[i] = block_itr[thread_id + i * ActualParam::BLOCK_THREADS];
        }
        BlockExchange(temp_storage).StripedToBlocked(items, items);
    }

    __device__ __forceinline__ void
    Load(const int32_t *block_itr,
         int32_t (&items)[ActualParam::THREAD_ITEMS],
         unsigned int valid_items,
         int32_t oob_default)
    {
#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            auto block_pos = thread_id + (i * ActualParam::BLOCK_THREADS);
            items[i] =
                (block_pos < valid_items) ? block_itr[block_pos] : oob_default;
        }
        BlockExchange(temp_storage).StripedToBlocked(items, items);
    }
};

class BlockStore {
private:
    struct _TempStorage : BlockExchange::TempStorage {
        volatile unsigned int valid_items;
    };
    _TempStorage &temp_storage;

    unsigned int thread_id;

    __device__ __forceinline__ void
    StoreDirectStriped(int32_t *block_itr,
                       int32_t (&items)[ActualParam::THREAD_ITEMS])
    {
        auto *thread_itr = block_itr + thread_id;

#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            thread_itr[(i * ActualParam::BLOCK_THREADS)] = items[i];
        }
    }

    __device__ __forceinline__ void
    StoreDirectStriped(int32_t *block_itr,
                       int32_t (&items)[ActualParam::THREAD_ITEMS],
                       unsigned int valid_items)
    {
        auto *thread_itr = block_itr + thread_id;

#pragma unroll
        for (size_t i = 0; i < ActualParam::THREAD_ITEMS; i++) {
            if ((i * ActualParam::BLOCK_THREADS) + thread_id < valid_items) {
                thread_itr[(i * ActualParam::BLOCK_THREADS)] = items[i];
            }
        }
    }

public:
    using TempStorage = Uninitialized<_TempStorage>;

    __device__ __forceinline__
    BlockStore(TempStorage &temp_storage)
        : temp_storage(temp_storage.Alias()),
          thread_id(threadIdx.x)
    {
    }

    __device__ __forceinline__ void
    Store(int32_t *block_itr, int32_t (&items)[ActualParam::THREAD_ITEMS])
    {
        BlockExchange(temp_storage).BlockedToStriped(items);
        StoreDirectStriped(block_itr, items);
    }

    __device__ __forceinline__ void
    Store(int32_t *block_itr,
          int32_t (&items)[ActualParam::THREAD_ITEMS],
          unsigned int valid_items)
    {
        BlockExchange(temp_storage).BlockedToStriped(items);
        if (thread_id == 0)
            temp_storage.valid_items = valid_items;

        __syncthreads();
        StoreDirectStriped(block_itr, items, temp_storage.valid_items);
    }
};

struct BlockAndWarpScans {
    static constexpr auto WARPS =
        (ActualParam::BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS;
    static constexpr auto STEPS = intlog2(WARP_THREADS);
    static constexpr auto MEMBER_MASK = 0xFFFFFFFFu;

    struct __align__(32) _TempStorage
    {
        int32_t warp_aggregates[WARPS];
        int32_t block_prefix;
    };

    using TempStorage = Uninitialized<_TempStorage>;

    _TempStorage &temp_storage;
    unsigned int thread_id;
    unsigned int warp_id;
    unsigned int lane_id;

    __device__ __forceinline__
    BlockAndWarpScans(TempStorage &temp_storage)
        : temp_storage(temp_storage.Alias()),
          thread_id(threadIdx.x),
          warp_id((WARPS == 1) ? 0 : thread_id / WARP_THREADS),
          lane_id(LaneId())
    {
    }

    static __device__ __forceinline__ void
    InclusiveScanShfl(int32_t input, int32_t &inclusive_output)
    {
        inclusive_output = input;

#pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++) {
            auto offset = 1 << STEP;

            asm volatile("{"
                         "  .reg .s32 r0;"
                         "  .reg .pred p;"
                         "  shfl.sync.up.b32 r0|p, %1, %2, 0, %4;"
                         "  @p add.s32 r0, r0, %3;"
                         "  mov.s32 %0, r0;"
                         "}"
                         : "=r"(inclusive_output)
                         : "r"(inclusive_output),
                           "r"(offset),
                           "r"(inclusive_output),
                           "r"(MEMBER_MASK));
        }
    }

    __device__ __forceinline__ int32_t
    ComputeWarpPrefix(int32_t warp_aggregate, int32_t &block_aggregate)
    {
        if (lane_id == WARP_THREADS - 1) {
            uninitialized_copy(temp_storage.warp_aggregates + warp_id,
                               warp_aggregate);
        }
        __syncthreads();

        int32_t warp_prefix;
        block_aggregate = temp_storage.warp_aggregates[0];

#pragma unroll
        for (size_t WARP = 1; WARP < WARPS; ++WARP) {
            if (warp_id == WARP)
                warp_prefix = block_aggregate;

            int32_t addend = temp_storage.warp_aggregates[WARP];
            block_aggregate += addend;
        }

        return warp_prefix;
    }

    __device__ __forceinline__ int32_t
    ComputeWarpPrefix(int32_t warp_aggregate,
                      int32_t &block_aggregate,
                      const int32_t &initial_value)
    {
        int32_t warp_prefix =
            ComputeWarpPrefix(warp_aggregate, block_aggregate);

        warp_prefix += initial_value;

        if (warp_id == 0)
            warp_prefix = initial_value;

        return warp_prefix;
    }

    __device__ __forceinline__ void
    ExclusiveScan(int32_t input,
                  int32_t &exclusive_output,
                  int32_t &block_aggregate)
    {
        int32_t inclusive_output;
        InclusiveScanShfl(input, inclusive_output);
        exclusive_output = inclusive_output - input;

        int32_t warp_prefix =
            ComputeWarpPrefix(inclusive_output, block_aggregate);

        if (warp_id != 0) {
            exclusive_output += warp_prefix;
            if (lane_id == 0)
                exclusive_output = warp_prefix;
        }
    }

    template <typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void
    ExclusiveScan(int32_t input,
                  int32_t &exclusive_output,
                  BlockPrefixCallbackOp &block_prefix_callback_op)
    {
        int32_t block_aggregate;
        ExclusiveScan(input, exclusive_output, block_aggregate);

        if (warp_id == 0) {
            int32_t block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0) {
                uninitialized_copy(&temp_storage.block_prefix, block_prefix);
                exclusive_output = block_prefix;
            }
        }
        __syncthreads();

        int32_t block_prefix = temp_storage.block_prefix;
        if (thread_id > 0) {
            exclusive_output += block_prefix;
        }
    }

    __device__ __forceinline__ void
    InclusiveScan(int32_t input, int32_t &inclusive_output)
    {
        int32_t block_aggregate;
        InclusiveScan(input, inclusive_output, block_aggregate);
    }
    __device__ __forceinline__ void
    InclusiveScan(int32_t input,
                  int32_t &inclusive_output,
                  int32_t &block_aggregate)
    {
        InclusiveScanShfl(input, inclusive_output);
        int32_t warp_prefix =
            ComputeWarpPrefix(inclusive_output, block_aggregate);
        if (warp_id != 0)
            inclusive_output += warp_prefix;
    }

    template <typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void
    InclusiveScan(int32_t input,
                  int32_t &exclusive_output,
                  BlockPrefixCallbackOp &block_prefix_callback_op)
    {
        int32_t block_aggregate;
        InclusiveScan(input, exclusive_output, block_aggregate);

        if (warp_id == 0) {
            int32_t block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0)
                uninitialized_copy(&temp_storage.block_prefix, block_prefix);
        }
        __syncthreads();

        int32_t block_prefix = temp_storage.block_prefix;
        exclusive_output += block_prefix;
    }
};

namespace helper {
template <int LENGTH>
__device__ __forceinline__ int32_t
ThreadScanInclusive(int32_t inclusive,
                    int32_t *input,
                    int32_t *output,
                    Int2Type<LENGTH>)
{
#pragma unroll
    for (int i = 0; i < LENGTH; ++i) {
        inclusive += input[i];
        output[i] = inclusive;
    }

    return inclusive;
}

template <size_t LENGTH>
__device__ __forceinline__ int32_t
ThreadScanInclusive(int32_t *input, int32_t *output)
{
    int32_t inclusive = input[0];
    output[0] = inclusive;

#pragma unroll
    for (size_t i = 1; i < LENGTH; ++i) {
        inclusive += input[i];
        output[i] = inclusive;
    }
    return inclusive;
}

template <int LENGTH>
__device__ __forceinline__ int32_t
ThreadScanInclusive(int32_t *input,
                    int32_t *output,
                    int32_t prefix,
                    bool apply_prefix = true)
{
    int32_t inclusive = input[0];
    if (apply_prefix)
        inclusive += prefix;

    output[0] = inclusive;

    return ThreadScanInclusive(inclusive,
                               input + 1,
                               output + 1,
                               Int2Type<LENGTH - 1>());
}

template <int LENGTH>
__device__ __forceinline__ int32_t
ThreadScanInclusive(int32_t (&input)[LENGTH],
                    int32_t (&output)[LENGTH],
                    int32_t prefix,
                    bool apply_prefix = true)
{
    return ThreadScanInclusive<LENGTH>((int32_t *)input,
                                       (int32_t *)output,
                                       prefix,
                                       apply_prefix);
}

template <int LENGTH, typename T>
__device__ __forceinline__ int32_t
ThreadReduce(T *input, int32_t prefix, Int2Type<LENGTH> /*length*/)
{
    int32_t retval = prefix;

#pragma unroll
    for (int i = 0; i < LENGTH; ++i)
        retval += input[i];

    return retval;
}

template <int LENGTH, typename T, typename AccumT = int32_t>
__device__ __forceinline__ AccumT
ThreadReduce(T *input, int32_t prefix)
{
    return ThreadReduce(input, prefix, Int2Type<LENGTH>());
}

template <int LENGTH>
__device__ __forceinline__ int32_t
ThreadReduce(int32_t *input)
{
    int32_t prefix = input[0];
    return ThreadReduce<LENGTH - 1>(input + 1, prefix);
}

template <int LENGTH>
__device__ __forceinline__ int32_t
ThreadReduce(int32_t (&input)[LENGTH], int32_t prefix)
{
    return ThreadReduce(input, prefix, Int2Type<LENGTH>());
}

template <int LENGTH>
__device__ __forceinline__ int32_t
ThreadReduce(int32_t (&input)[LENGTH])
{
    return ThreadReduce<LENGTH>((int32_t *)input);
}

} // namespace helper

class TilePrefixCallbackOp {
    static constexpr auto STEPS = intlog2(WARP_THREADS);
    static constexpr auto MEMBER_MASK = 0xFFFFFFFFu;

    struct _TempStorage {
        int32_t exclusive_prefix;
        int32_t inclusive_prefix;
        int32_t block_aggregate;
    };

    _TempStorage &temp_storage;
    ScanTileState &tile_status;
    int tile_idx;
    int32_t exclusive_prefix;
    int32_t inclusive_prefix;
    unsigned int lane_id;

    __device__ __forceinline__ unsigned int
    SHFL_DOWN_SYNC(unsigned int word,
                   int src_offset,
                   int flags,
                   unsigned int member_mask)
    {
        asm volatile(
            "shfl.sync.down.b32 %0, %1, %2, %3, %4;"
            : "=r"(word)
            : "r"(word), "r"(src_offset), "r"(flags), "r"(member_mask));
        return word;
    }

    __device__ __forceinline__ unsigned int
    LaneMaskGe()
    {
        unsigned int ret;
        asm("mov.u32 %0, %%lanemask_ge;" : "=r"(ret));
        return ret;
    }

    __device__ __forceinline__ int32_t
    ReduceStep(int32_t input, int last_lane, int offset)
    {
        auto output = input;
        typedef typename UnitWord<int32_t>::ShuffleWord ShuffleWord;

        static constexpr int WORDS =
            (sizeof(int32_t) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

        int32_t temp;
        ShuffleWord *output_alias = reinterpret_cast<ShuffleWord *>(&temp);
        ShuffleWord *input_alias = reinterpret_cast<ShuffleWord *>(&input);

        unsigned int shuffle_word;
        shuffle_word = SHFL_DOWN_SYNC((unsigned int)input_alias[0],
                                      offset,
                                      last_lane,
                                      MEMBER_MASK);
        output_alias[0] = shuffle_word;

#pragma unroll
        for (int WORD = 1; WORD < WORDS; ++WORD) {
            shuffle_word = SHFL_DOWN_SYNC((unsigned int)input_alias[WORD],
                                          offset,
                                          last_lane,
                                          MEMBER_MASK);
            output_alias[WORD] = shuffle_word;
        }

        // Perform reduction op if valid
        if (offset + lane_id <= last_lane)
            output = input + temp;

        return output;
    }

public:
    using TempStorage = Uninitialized<_TempStorage>;
    using StatusWord = typename ScanTileState::StatusWord;

    __device__ __forceinline__
    TilePrefixCallbackOp(ScanTileState &tile_status,
                         TempStorage &temp_storage,
                         int tile_idx)
        : temp_storage(temp_storage.Alias()),
          tile_status(tile_status),
          tile_idx(tile_idx),
          lane_id(LaneId())
    {
    }

    __device__ __forceinline__ void
    ProcessWindow(size_t predecessor_idx,
                  StatusWord &predecessor_status,
                  int32_t &window_aggregate)
    {
        int32_t value;
        tile_status.WaitForValid(predecessor_idx, predecessor_status, value);

        int tail_flag = (predecessor_status == StatusWord(SCAN_TILE_INCLUSIVE));
        auto warp_flags = __ballot_sync(MEMBER_MASK, tail_flag);

        warp_flags &= LaneMaskGe();
        warp_flags |= 1u << (WARP_THREADS - 1);

        auto last_lane = __clz(static_cast<int>(__brev(warp_flags)));
        window_aggregate = value;

#pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++) {
            window_aggregate =
                ReduceStep(window_aggregate, last_lane, 1 << STEP);
        }
    }

    __device__ __forceinline__ int32_t
    operator()(int32_t block_aggregate)
    {
        if (threadIdx.x == 0) {
            uninitialized_copy(&temp_storage.block_aggregate, block_aggregate);
            tile_status.SetPartial(tile_idx, block_aggregate);
        }

        auto predecessor_idx = tile_idx - threadIdx.x - 1;
        StatusWord predecessor_status;
        int32_t window_aggregate;

        delay<450>();
        ProcessWindow(predecessor_idx, predecessor_status, window_aggregate);
        exclusive_prefix = window_aggregate;
        while (__all_sync(
            0xffffffff,
            (predecessor_status != StatusWord(SCAN_TILE_INCLUSIVE)))) {
            predecessor_idx -= WARP_THREADS;

            ProcessWindow(predecessor_idx,
                          predecessor_status,
                          window_aggregate);
            exclusive_prefix = window_aggregate + exclusive_prefix;
        }

        if (threadIdx.x == 0) {
            inclusive_prefix = exclusive_prefix + block_aggregate;
            tile_status.SetInclusive(tile_idx, inclusive_prefix);
            uninitialized_copy(&temp_storage.exclusive_prefix,
                               exclusive_prefix);
            uninitialized_copy(&temp_storage.inclusive_prefix,
                               inclusive_prefix);
        }

        return exclusive_prefix;
    }
};

class BlockScan {
private:
    typedef typename BlockAndWarpScans::TempStorage _TempStorage;

    _TempStorage &temp_storage;

    unsigned int linear_tid;

public:
    using TempStorage = Uninitialized<_TempStorage>;

    __device__ __forceinline__
    BlockScan(TempStorage &temp_storage)
        : temp_storage(temp_storage.Alias()),
          linear_tid(threadIdx.x)
    {
    }

    template <typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void
    ExclusiveScan(int32_t input,
                  int32_t &output,
                  BlockPrefixCallbackOp &block_prefix_callback_op)

    {
        BlockAndWarpScans(temp_storage)
            .ExclusiveScan(input, output, block_prefix_callback_op);
    }

    __device__ __forceinline__ void
    ExclusiveScanAgg(int32_t input, int32_t &output, int32_t &block_aggregate)
    {
        BlockAndWarpScans(temp_storage)
            .ExclusiveScan(input, output, block_aggregate);
    }

    template <typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void
    InclusiveScan(int32_t input,
                  int32_t &output,
                  BlockPrefixCallbackOp &block_prefix_callback_op)

    {
        BlockAndWarpScans(temp_storage)
            .InclusiveScan(input, output, block_prefix_callback_op);
    }

    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void
    InclusiveScan(int32_t (&input)[ITEMS_PER_THREAD],
                  int32_t (&output)[ITEMS_PER_THREAD],
                  int32_t &block_aggregate)
    {
        if (ITEMS_PER_THREAD == 1) {
            InclusiveScan(input[0], output[0], block_aggregate);
        } else {
            int32_t thread_prefix = helper::ThreadReduce(input);
            ExclusiveScanAgg(thread_prefix, thread_prefix, block_aggregate);
            helper::ThreadScanInclusive(input,
                                        output,
                                        thread_prefix,
                                        (linear_tid != 0));
        }
    }

    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void
    InclusiveScan(int32_t (&input)[ITEMS_PER_THREAD],
                  int32_t (&output)[ITEMS_PER_THREAD],
                  TilePrefixCallbackOp &block_prefix_callback_op)

    {
        if (ITEMS_PER_THREAD == 1) {
            InclusiveScan(input[0], output[0], block_prefix_callback_op);
        } else {
            // Reduce consecutive thread items in registers
            int32_t thread_prefix = helper::ThreadReduce(input);

            // Exclusive thread block-scan
            ExclusiveScan(thread_prefix,
                          thread_prefix,
                          block_prefix_callback_op);

            // Inclusive scan in registers with prefix as seed
            helper::ThreadScanInclusive(input, output, thread_prefix);
        }
    }
};

struct AgentScan {
    static constexpr auto TILE_ITEMS =
        ActualParam::BLOCK_THREADS * ActualParam::THREAD_ITEMS;

    union _TempStorage {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;

        struct ScanStorage {
            typename TilePrefixCallbackOp::TempStorage prefix;
            typename BlockScan::TempStorage scan;
        } scan_storage;
    };

    using TempStorage = Uninitialized<_TempStorage>;

    _TempStorage &temp_storage;
    const int32_t *const d_in;
    int32_t *d_out;
    Sum scan_op{};

    __device__ __forceinline__ void
    ScanTile(int32_t (&items)[ActualParam::THREAD_ITEMS],
             int32_t &block_aggregate)
    {
        BlockScan(temp_storage.scan_storage.scan)
            .InclusiveScan(items, items, block_aggregate);
    }

    __device__ __forceinline__ void
    ScanTile(int32_t (&items)[ActualParam::THREAD_ITEMS],
             TilePrefixCallbackOp &prefix_op)
    {
        BlockScan(temp_storage.scan_storage.scan)
            .InclusiveScan(items, items, prefix_op);
    }

    __device__ __forceinline__
    AgentScan(TempStorage &temp_storage, const int32_t *d_in, int32_t *d_out)
        : temp_storage(temp_storage.Alias()),
          d_in(d_in),
          d_out(d_out)
    {
    }

    template <bool IS_LAST_TILE>
    __device__ __forceinline__ void
    ConsumeTile(int num_remaining,
                int tile_idx,
                int tile_offset,
                ScanTileState &tile_state)
    {
        int32_t items[ActualParam::THREAD_ITEMS];

        if (IS_LAST_TILE) {
            BlockLoad(temp_storage.load)
                .Load(d_in + tile_offset,
                      items,
                      num_remaining,
                      *(d_in + tile_offset));
        } else {
            BlockLoad(temp_storage.load).Load(d_in + tile_offset, items);
        }

        __syncthreads();

        if (tile_idx == 0) {
            int32_t block_aggregate;
            ScanTile(items, block_aggregate);

            if ((!IS_LAST_TILE) && (threadIdx.x == 0)) {
                tile_state.SetInclusive(0, block_aggregate);
            }
        } else {
            TilePrefixCallbackOp prefix_op(tile_state,
                                           temp_storage.scan_storage.prefix,
                                           tile_idx);
            ScanTile(items, prefix_op);
        }

        __syncthreads();

        if (IS_LAST_TILE) {
            BlockStore(temp_storage.store)
                .Store(d_out + tile_offset, items, num_remaining);
        } else {
            BlockStore(temp_storage.store).Store(d_out + tile_offset, items);
        }
    }

    __device__ __forceinline__ void
    ConsumeRange(int num_items, ScanTileState &tile_state, int start_tile)
    {
        auto tile_idx = start_tile + blockIdx.x;
        auto tile_offset = int(TILE_ITEMS) * tile_idx;

        int num_remaining = num_items - tile_offset;

        if (num_remaining > TILE_ITEMS) {
            ConsumeTile<false>(num_remaining,
                               tile_idx,
                               tile_offset,
                               tile_state);
        } else if (num_remaining > 0) {
            ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
        }
    }
};

template <int ALLOCATIONS>
__host__ __forceinline__ cudaError_t
WorkingArea(void *memory_pool,
            size_t &memory_size,
            void *(&allocations)[ALLOCATIONS],
            size_t (&allocation_sizes)[ALLOCATIONS])
{
    const int ALIGN_BYTES = 256;
    const int ALIGN_MASK = ~(ALIGN_BYTES - 1);

    size_t allocation_offsets[ALLOCATIONS];
    size_t bytes_needed = 0;
    for (int i = 0; i < ALLOCATIONS; ++i) {
        size_t allocation_bytes =
            (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
        allocation_offsets[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }
    bytes_needed += ALIGN_BYTES - 1;

    if (!memory_pool) {
        memory_size = bytes_needed;
        return cudaSuccess;
    }

    if (memory_size < bytes_needed) {
        return check_cuda_error(cudaErrorInvalidValue);
    }

    memory_pool = (void *)((size_t(memory_pool) + ALIGN_BYTES - 1) &
                           static_cast<size_t>(ALIGN_MASK));
    for (int i = 0; i < ALLOCATIONS; ++i) {
        allocations[i] =
            static_cast<char *>(memory_pool) + allocation_offsets[i];
    }

    return cudaSuccess;
}

__global__ void
init(ScanTileState tile_state, int num_tiles)
{
    tile_state.InitializeStatus(num_tiles);
}

__launch_bounds__(ActualParam::BLOCK_THREADS) __global__
    void main_kernel(const int32_t *d_in,
                     int32_t *d_out,
                     ScanTileState tile_state,
                     int start_tile,
                     int num_items)
{
    // Shared memory for AgentScan
    __shared__ typename AgentScan::TempStorage temp_storage;

    // Process tiles
    AgentScan(temp_storage, d_in, d_out)
        .ConsumeRange(num_items, tile_state, start_tile);
}

__host__ cudaError_t
InclusiveSum(void *working_area,
             size_t &temp_storage_bytes,
             const int32_t *d_in,
             int32_t *d_out,
             int num_items,
             cudaStream_t stream = 0)
{
    static constexpr auto INIT_KERNEL_THREADS = 128;
    constexpr auto ret = cudaSuccess;

    auto tile_size = ActualParam::BLOCK_THREADS * ActualParam::THREAD_ITEMS;
    auto num_tiles = (num_items + tile_size - 1) / tile_size;

    size_t allocation_sizes[1] = {ScanTileState::AllocationSize(num_tiles)};

    void *allocations[1] = {};
    check_cuda_error(WorkingArea(working_area,
                                 temp_storage_bytes,
                                 allocations,
                                 allocation_sizes));
    if (working_area == NULL) {
        return cudaErrorMemoryAllocation;
    }

    if (num_items == 0)
        return ret;

    ScanTileState tile_state;
    check_cuda_error(
        tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));

    auto init_grid_size = static_cast<unsigned int>(
        (num_tiles + INIT_KERNEL_THREADS - 1) / INIT_KERNEL_THREADS);

    init<<<init_grid_size, INIT_KERNEL_THREADS, 0, stream>>>(tile_state,
                                                             num_tiles);
    check_cuda_error(cudaPeekAtLastError());

    int max_dim_x;
    check_cuda_error(
        cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, 0));

    auto scan_grid_size =
        static_cast<unsigned int>(mixedMin(num_tiles, max_dim_x));
    for (size_t start_tile = 0; start_tile < num_tiles;
         start_tile += scan_grid_size) {
        main_kernel<<<scan_grid_size, ActualParam::BLOCK_THREADS, 0, stream>>>(
            d_in,
            d_out,
            tile_state,
            start_tile,
            num_items);
        check_cuda_error(cudaPeekAtLastError());
    }
    return cudaSuccess;
}

/**
 * Implement your CUDA inclusive scan here. Feel free to add helper functions,
 * kernels or allocate temporary memory. However, you must not modify other
 * files. CAUTION: make sure you synchronize your kernels properly and free all
 * allocated memory.
 *
 * @param d_input: input array on device
 * @param d_output: output array on device
 * @param size: number of elements in the input array
 */
void
impl_nvidia_decoupled_lookback(const int32_t *d_input,
                               int32_t *d_output,
                               size_t size)
{
    void *working_area = nullptr;
    size_t working_area_size = 0;
    auto num_items = static_cast<int>(size);
    auto tile_size = ActualParam::BLOCK_THREADS * ActualParam::THREAD_ITEMS;
    auto num_tiles = (size + tile_size - 1) / tile_size;
    size_t allocation_sizes[1] = {ScanTileState::AllocationSize(num_tiles)};
    void *allocations[1] = {};
    check_cuda_error(WorkingArea(working_area,
                                 working_area_size,
                                 allocations,
                                 allocation_sizes));
    cudaMalloc(&working_area, working_area_size);
    InclusiveSum(working_area, working_area_size, d_input, d_output, num_items);

    cudaFree(working_area);
}
