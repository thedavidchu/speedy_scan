// adapted from nvidia's CUB implementation

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda/std/functional>
#include <type_traits>
#include <vector_types.h>

static constexpr unsigned int WARP_THREADS = 32;
#ifdef BOIS
static constexpr unsigned int NUM_ITEMS_PER_THREAD = BOIS;
#else
static constexpr unsigned int NUM_ITEMS_PER_THREAD = 15;
#endif

#ifdef BOYS
static constexpr unsigned int THREADS_PER_BLOCK = BOYS;
#else
static constexpr unsigned int THREADS_PER_BLOCK = 128;
#endif

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

__host__ __device__ constexpr bool
powerOfTwo(int n)
{
    return (n & (n - 1)) == 0;
}

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
#ifdef NO_ASM_MAGIC
    __stcg(ptr, val);
#else
    asm volatile("st.relaxed.gpu.v2.u32 [%0], {%1, %2};"
                 :
                 : "l"(ptr), "r"(val.x), "r"(val.y)
                 : "memory");
#endif
}
static __device__ __forceinline__ uint2
load_relaxed(uint2 const *ptr)
{
    uint2 retval;
#ifdef NO_ASM_MAGIC
    retval = __ldcv(ptr);
#else

    asm volatile("ld.relaxed.gpu.v2.u32 {%0, %1}, [%2];"
                 : "=r"(retval.x), "=r"(retval.y)
                 : "l"(ptr)
                 : "memory");
#endif
    return retval;
}

__device__ __forceinline__ unsigned int
LaneId()
{
    unsigned int ret;
#ifdef NO_ASM_MAGIC
    ret = threadIdx.x & 31;
#else
    asm("mov.u32 %0, %%laneid;" : "=r"(ret));
#endif
    return ret;
}

__device__ __forceinline__ unsigned int
LaneMaskGe()
{
    unsigned int ret;
#ifdef NO_ASM_MAGIC
    ret = ~((1u << LaneId()) - 1);
#else
    asm("mov.u32 %0, %%lanemask_ge;" : "=r"(ret));
#endif
    return ret;
}

__device__ __forceinline__ unsigned int
SHFL_DOWN_SYNC(unsigned int word,
               unsigned int src_offset,
               int flags,
               unsigned int member_mask)
{
#ifdef NO_ASM_MAGIC
    word = __shfl_down_sync(member_mask, word, src_offset, WARP_THREADS);
#else
    asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;"
                 : "=r"(word)
                 : "r"(word), "r"(src_offset), "r"(flags), "r"(member_mask));
#endif
    return word;
}

enum ScanTileStatus {
    SCAN_TILE_OOB,          // Out-of-bounds (e.g., padding)
    SCAN_TILE_INVALID = 99, // Not yet processed
    SCAN_TILE_PARTIAL,      // Tile aggregate is available
    SCAN_TILE_INCLUSIVE,    // Inclusive tile prefix is available
};

struct ScanTileState {
    using StatusWord = unsigned int;
    using TxnWord = uint2;
    static constexpr auto TILE_STATUS_PADDING = 32u;

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
    Init(void *d_temp_storage)
    {
        d_tile_descriptors = reinterpret_cast<TxnWord *>(d_temp_storage);
        return cudaSuccess;
    }

    __device__ __forceinline__ void
    InitializeStatus(size_t num_tiles)
    {
        auto tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

        TxnWord val;
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
    SetInclusive(unsigned int tile_idx, int32_t tile_inclusive)
    {
        TileDescriptor tile_descriptor{SCAN_TILE_INCLUSIVE, tile_inclusive};

        TxnWord alias;
        *reinterpret_cast<TileDescriptor *>(&alias) = tile_descriptor;

        store_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx,
                      alias);
    }

    __device__ __forceinline__ void
    SetPartial(unsigned int tile_idx, int32_t tile_partial)
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
        TxnWord alias =
            load_relaxed(d_tile_descriptors + TILE_STATUS_PADDING + tile_idx);
        TileDescriptor tile_descriptor =
            reinterpret_cast<TileDescriptor &>(alias);

        while (__any_sync(0xffffffff,
                          (tile_descriptor.status == SCAN_TILE_INVALID))) {
            delay<350>();
            alias = load_relaxed(d_tile_descriptors + TILE_STATUS_PADDING +
                                 tile_idx);
            tile_descriptor = reinterpret_cast<TileDescriptor &>(alias);
        }

        status = tile_descriptor.status;
        value = tile_descriptor.value;
    }
};

/**
 * \brief Enumeration of cache modifiers for memory load operations.
 */
enum CacheLoadModifier {
    LOAD_DEFAULT, ///< Default (no modifier)
};

struct ScanPolicy {
    static constexpr auto THREAD_ITEMS = std::max(
        1u,
        std::min(NUM_ITEMS_PER_THREAD * 4 / static_cast<int>(sizeof(int32_t)),
                 NUM_ITEMS_PER_THREAD * 2));
    static constexpr auto BLOCK_THREADS = mixedMin(
        THREADS_PER_BLOCK,
        (((1024 * 48) / (sizeof(int32_t) * THREAD_ITEMS)) + 31) / 32 * 32);
    static constexpr CacheLoadModifier LOAD_MODIFIER = LOAD_DEFAULT;
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

/**
 * \brief A simple "NULL" marker type
 */
struct NullType {
    using value_type = NullType;

    template <typename T>
    __host__ __device__ __forceinline__ NullType &
    operator=(const T &)
    {
        return *this;
    }

    __host__ __device__ __forceinline__ bool
    operator==(const NullType &)
    {
        return true;
    }

    __host__ __device__ __forceinline__ bool
    operator!=(const NullType &)
    {
        return false;
    }
};

template <typename T> struct AlignBytes {
    struct Pad {
        T val;
        char byte;
    };

    static constexpr auto ALIGN_BYTES = sizeof(Pad) - sizeof(T);
    using Type = T;
};
template <typename T> struct AlignBytes<volatile T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const volatile T> : AlignBytes<T> {};

template <typename T> struct UnitWord {
    static constexpr auto ALIGN_BYTES = AlignBytes<T>::ALIGN_BYTES;

    template <typename Unit> struct IsMultiple {
        static constexpr auto UNIT_ALIGN_BYTES = AlignBytes<Unit>::ALIGN_BYTES;
        static constexpr auto IS_MULTIPLE =
            (sizeof(T) % sizeof(Unit) == 0) &&
            (int(ALIGN_BYTES) % int(UNIT_ALIGN_BYTES) == 0);
    };

    /// Biggest shuffle word that T is a whole multiple of and is not larger
    /// than the alignment of T
    using ShuffleWord =
        std::conditional_t<IsMultiple<int>::IS_MULTIPLE,
                           unsigned int,
                           std::conditional_t<IsMultiple<short>::IS_MULTIPLE,
                                              unsigned short,
                                              unsigned char>>;

    /// Biggest volatile word that T is a whole multiple of and is not larger
    /// than the alignment of T
    using VolatileWord = std::conditional_t<IsMultiple<long long>::IS_MULTIPLE,
                                            unsigned long long,
                                            ShuffleWord>;

    /// Biggest memory-access word that T is a whole multiple of and is not
    /// larger than the alignment of T
    using DeviceWord = std::conditional_t<IsMultiple<longlong2>::IS_MULTIPLE,
                                          ulonglong2,
                                          VolatileWord>;
};
template <typename T> struct UnitWord<volatile T> : UnitWord<T> {};
template <typename T> struct UnitWord<const T> : UnitWord<T> {};
template <typename T> struct UnitWord<const volatile T> : UnitWord<T> {};

template <typename T> struct Uninitialized {
    /// Biggest memory-access word that T is a whole multiple of and is not
    /// larger than the alignment of T
    using DeviceWord = typename UnitWord<T>::DeviceWord;

    static constexpr std::size_t DATA_SIZE = sizeof(T);
    static constexpr std::size_t WORD_SIZE = sizeof(DeviceWord);
    static constexpr std::size_t WORDS = DATA_SIZE / WORD_SIZE;

    /// Backing storage
    DeviceWord storage[WORDS];

    /// Alias
    __host__ __device__ __forceinline__ T &
    Alias()
    {
        return reinterpret_cast<T &>(*this);
    }
};

class BlockExchange {
private:
    /******************************************************************************
     * Constants
     ******************************************************************************/

    /// Constants
    static constexpr auto TIME_SLICED_ITEMS =
        THREADS_PER_BLOCK * ScanPolicy::THREAD_ITEMS;

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Shared memory storage layout type
    struct __align__(16) _TempStorage { int32_t buff[TIME_SLICED_ITEMS]; };

public:
    /// \smemstorage{BlockExchange}
    using TempStorage = Uninitialized<_TempStorage>;

private:
    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;
    unsigned int lane_id;

public:
    __device__ __forceinline__
    BlockExchange(
        TempStorage &temp_storage) ///< [in] Reference to memory allocation
                                   ///< having layout type TempStorage
        : temp_storage(temp_storage.Alias()),
          linear_tid(threadIdx.x),
          lane_id(LaneId())
    {
    }

    __device__ __forceinline__ void
    StripedToBlocked(int32_t (&items)[ScanPolicy::THREAD_ITEMS])
    {
#pragma unroll
        for (unsigned int i = 0; i < ScanPolicy::THREAD_ITEMS; i++) {
            auto item_offset = i * THREADS_PER_BLOCK + linear_tid;
            temp_storage.buff[item_offset] = items[i];
        }

        __syncthreads();

#pragma unroll
        for (unsigned int i = 0; i < ScanPolicy::THREAD_ITEMS; i++) {
            auto item_offset = linear_tid * ScanPolicy::THREAD_ITEMS + i;
            items[i] = temp_storage.buff[item_offset];
        }
    }

    __device__ __forceinline__ void
    BlockedToStriped(int32_t (&items)[ScanPolicy::THREAD_ITEMS])
    {
#pragma unroll
        for (unsigned int i = 0; i < ScanPolicy::THREAD_ITEMS; i++) {
            auto item_offset = linear_tid * ScanPolicy::THREAD_ITEMS + i;
            temp_storage.buff[item_offset] = items[i];
        }

        __syncthreads();

#pragma unroll
        for (unsigned int i = 0; i < ScanPolicy::THREAD_ITEMS; i++) {
            auto item_offset = i * THREADS_PER_BLOCK + linear_tid;
            items[i] = temp_storage.buff[item_offset];
        }
    }
};

class BlockLoad {
public:
    using TempStorage = BlockExchange::TempStorage;

private:
    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Thread reference to shared storage
    TempStorage &temp_storage;

    /// Linear thread-id
    unsigned long linear_tid;

public:
    /**
     * \name Collective constructors
     *********************************************************************/

    __device__ __forceinline__
    BlockLoad(
        TempStorage &temp_storage) ///< [in] Reference to memory allocation
                                   ///< having layout type TempStorage
        : temp_storage(temp_storage),
          linear_tid(threadIdx.x)
    {
    }

    __device__ __forceinline__ void
    Load(const int32_t *block_itr, ///< [in] The thread block's base input
                                   ///< iterator for loading from
         int32_t (&items)[ScanPolicy::THREAD_ITEMS]) ///< [out] Data to load
    {
/// Load a linear segment of items from memory
#pragma unroll
        for (size_t ITEM = 0; ITEM < ScanPolicy::THREAD_ITEMS; ITEM++) {
            items[ITEM] =
                block_itr[linear_tid + ITEM * ScanPolicy::BLOCK_THREADS];
        }
        BlockExchange(temp_storage).StripedToBlocked(items);
    }

    __device__ __forceinline__ void
    Load(const int32_t *block_itr, ///< [in] The thread block's base input
                                   ///< iterator for loading from
         int32_t (&items)[ScanPolicy::THREAD_ITEMS], ///< [out] Data to load
         const size_t valid_items, ///< [in] Number of valid items to load
         const int32_t
             oob_default) ///< [in] Default value to assign out-of-bound items
    {
/// Load a linear segment of items from memory, guarded by range, with a
/// fall-back assignment of out-of-bound elements
#pragma unroll
        for (size_t ITEM = 0; ITEM < ScanPolicy::THREAD_ITEMS; ITEM++) {
            auto block_pos = linear_tid + ITEM * ScanPolicy::BLOCK_THREADS;
            items[ITEM] =
                (block_pos < valid_items) ? block_itr[block_pos] : oob_default;
        }
        BlockExchange(temp_storage).StripedToBlocked(items);
    }
};

class BlockStore {
private:
    struct _TempStorage : BlockExchange::TempStorage {
        volatile size_t valid_items;
    };
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;

public:
    using TempStorage = Uninitialized<_TempStorage>;

    /**
     * \brief Collective constructor using the specified memory allocation as
     * temporary storage.
     */
    __device__ __forceinline__
    BlockStore(
        TempStorage &temp_storage) ///< [in] Reference to memory allocation
                                   ///< having layout type TempStorage
        : temp_storage(temp_storage.Alias()),
          linear_tid(threadIdx.x)
    {
    }

    __device__ __forceinline__ void
    Store(int32_t *block_itr, ///< [out] The thread block's base output iterator
                              ///< for storing to
          int32_t (&items)[ScanPolicy::THREAD_ITEMS]) ///< [in] Data to store
    {
        BlockExchange(temp_storage).BlockedToStriped(items);

        /**
         * \brief Store a striped arrangement of data across the thread block
         * into a linear segment of items.
         */
        int32_t *thread_itr = block_itr + linear_tid;

// Store directly in striped order
#pragma unroll
        for (size_t ITEM = 0; ITEM < ScanPolicy::THREAD_ITEMS; ITEM++) {
            thread_itr[ITEM * ScanPolicy::BLOCK_THREADS] = items[ITEM];
        }
    }

    __device__ __forceinline__ void
    Store(int32_t *block_itr, ///< [out] The thread block's base output iterator
                              ///< for storing to
          int32_t (&items)[ScanPolicy::THREAD_ITEMS], ///< [in] Data to store
          size_t valid_items) ///< [in] Number of valid items to write
    {
        BlockExchange(temp_storage).BlockedToStriped(items);
        if (linear_tid == 0)
            temp_storage.valid_items =
                valid_items; // Move through volatile smem as a workaround to
                             // prevent RF spilling on subsequent loads
        __syncthreads();

        /**
         * \brief Store a striped arrangement of data across the thread block
         * into a linear segment of items, guarded by range
         */
        int32_t *thread_itr = block_itr + linear_tid;

// Store directly in striped order
#pragma unroll
        for (size_t ITEM = 0; ITEM < ScanPolicy::THREAD_ITEMS; ITEM++) {
            if ((ITEM * ScanPolicy::BLOCK_THREADS) + linear_tid < valid_items) {
                thread_itr[(ITEM * ScanPolicy::BLOCK_THREADS)] = items[ITEM];
            }
        }
    }
};

__host__ __device__ __forceinline__ constexpr int
intlog2(int n)
{
    /// Static logarithm value
    int ret = 0;
    while (n > 1) {
        n >>= 1;
        ++ret;
    }
    return ret;
}

class WarpReduce {
private:
    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/
    static constexpr auto STEPS = intlog2(WARP_THREADS);
    static constexpr auto MEMBER_MASK = 0xFFFFFFFFu;

    /// Shared memory storage layout type for WarpReduce
    using _TempStorage = NullType;

    /******************************************************************************
     * Thread fields
     ******************************************************************************/
    unsigned int lane_id;

    /******************************************************************************
     * Utility methods
     ******************************************************************************/
    __device__ __forceinline__ int32_t
    ReduceStep(int32_t input,       ///< [in] Calling thread's input item.
               int last_lane,       ///< [in] Index of last lane in segment
               unsigned int offset) ///< [in] Up-offset to pull from
    {
        auto output = input;
        using ShuffleWord = UnitWord<int32_t>::ShuffleWord;

        static constexpr auto WORDS =
            (sizeof(int32_t) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

        int32_t temp;
        ShuffleWord *output_alias = reinterpret_cast<ShuffleWord *>(&temp);
        ShuffleWord *input_alias = reinterpret_cast<ShuffleWord *>(&input);

        unsigned int shuffle_word;
        shuffle_word = SHFL_DOWN_SYNC((unsigned int)input_alias[0],
                                      offset,
                                      last_lane,
                                      MEMBER_MASK);
        *output_alias = shuffle_word;

#pragma unroll
        for (unsigned int WORD = 1; WORD < WORDS; ++WORD) {
            shuffle_word = SHFL_DOWN_SYNC((unsigned int)input_alias[WORD],
                                          offset,
                                          last_lane,
                                          MEMBER_MASK);
            output_alias[WORD] = shuffle_word;
        }

        // Perform reduction op if valid
        if (int(offset + lane_id) <= last_lane)
            output = input + temp;

        return output;
    }

public:
    /// \smemstorage{WarpReduce}
    using TempStorage = Uninitialized<_TempStorage>;

    __device__ __forceinline__
    WarpReduce()
        : lane_id(LaneId())
    {
    }

    template <typename FlagT>
    __device__ __forceinline__ int32_t
    TailSegmentedReduce(
        int32_t input,   ///< [in] Calling thread's input
        FlagT tail_flag) ///< [in] Tail flag denoting whether or not \p input is
                         ///< the end of the current segment
    {
        // Identify all predecessors that are inclusive
        auto warp_flags = __ballot_sync(MEMBER_MASK, tail_flag);

        // Mask out the bits below the current thread
        warp_flags &= LaneMaskGe();

        // Mask in the last lane of logical warp
        warp_flags |= 1u << (WARP_THREADS - 1);

        // Find the next set flag
        auto last_lane = __clz(int(__brev(warp_flags)));

        auto output = input;

#pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++) {
            output = ReduceStep(output, last_lane, 1u << STEP);
        }
        return output;
    }
};
struct TilePrefixCallbackOp {

    // Temporary storage type
    struct _TempStorage {
        using warp_reduce = WarpReduce::TempStorage;
        int32_t exclusive_prefix;
        int32_t inclusive_prefix;
        int32_t block_aggregate;
    };

    using TempStorage = Uninitialized<_TempStorage>;

    // Type of status word
    using StatusWord = ScanTileState::StatusWord;

    // Fields
    _TempStorage &temp_storage; ///< Reference to a warp-reduction instance
    ScanTileState &tile_status; ///< Interface to tile status
    unsigned int tile_idx;      ///< The current tile index
    int32_t exclusive_prefix;   ///< Exclusive prefix for the tile
    int32_t inclusive_prefix;   ///< Inclusive prefix for the tile

    // Constructor
    __device__ __forceinline__
    TilePrefixCallbackOp(ScanTileState &tile_status,
                         TempStorage &temp_storage,
                         unsigned int tile_idx)
        : temp_storage(temp_storage.Alias()),
          tile_status(tile_status),
          tile_idx(tile_idx)
    {
    }

    // Block until all predecessors within the warp-wide window have non-invalid
    // status
    __device__ __forceinline__ void
    ProcessWindow(
        int predecessor_idx,            ///< Preceding tile index to inspect
        StatusWord &predecessor_status, ///< [out] Preceding tile status
        int32_t &window_aggregate) ///< [out] Relevant partial reduction from
                                   ///< this window of preceding tiles
    {
        int32_t value;

        tile_status.WaitForValid(predecessor_idx, predecessor_status, value);

        // Perform a segmented reduction to get the prefix for the current
        // window. Use the swizzled scan operator because we are now scanning
        // *down* towards thread0.

        int tail_flag = (predecessor_status == StatusWord(SCAN_TILE_INCLUSIVE));
        window_aggregate = WarpReduce().TailSegmentedReduce(value, tail_flag);
    }

    // BlockScan prefix callback functor (called by the first warp)
    __device__ __forceinline__ int32_t
    operator()(int32_t block_aggregate)
    {

        // Update our status with our tile-aggregate
        if (threadIdx.x == 0) {
            temp_storage.block_aggregate = block_aggregate;

            tile_status.SetPartial(tile_idx, block_aggregate);
        }

        int predecessor_idx = int(tile_idx) - int(threadIdx.x) - 1;
        StatusWord predecessor_status;
        int32_t window_aggregate;

        // Wait for the warp-wide window of predecessor tiles to become valid
        delay<450>();
        ProcessWindow(predecessor_idx, predecessor_status, window_aggregate);

        // The exclusive tile prefix starts out as the current window aggregate
        exclusive_prefix = window_aggregate;

        // Keep sliding the window back until we come across a tile whose
        // inclusive prefix is known
        while (__all_sync(
            0xffffffff,
            (predecessor_status != StatusWord(SCAN_TILE_INCLUSIVE)))) {
            predecessor_idx -= WARP_THREADS;

            // Update exclusive tile prefix with the window prefix
            ProcessWindow(predecessor_idx,
                          predecessor_status,
                          window_aggregate);
            exclusive_prefix += window_aggregate;
        }

        // Compute the inclusive tile prefix and update the status for this tile
        if (threadIdx.x == 0) {
            inclusive_prefix = exclusive_prefix + block_aggregate;
            tile_status.SetInclusive(tile_idx, inclusive_prefix);

            temp_storage.exclusive_prefix = exclusive_prefix;

            temp_storage.inclusive_prefix = inclusive_prefix;
        }

        // Return exclusive_prefix
        return exclusive_prefix;
    }
};

struct BlockScanWarpScans {
    static constexpr auto WARPS =
        (THREADS_PER_BLOCK + WARP_THREADS - 1) / WARP_THREADS;
    static constexpr auto STEPS = intlog2(WARP_THREADS);
    static constexpr auto MEMBER_MASK = 0xFFFFFFFFu;

    struct __align__(32) _TempStorage
    {
        int32_t warp_aggregates[WARPS];
        int32_t block_prefix; ///< Shared prefix for the entire thread block
    };

    /// Alias wrapper allowing storage to be unioned
    using TempStorage = Uninitialized<_TempStorage>;

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    // Thread fields
    _TempStorage &temp_storage;
    unsigned int linear_tid;
    unsigned int warp_id;
    unsigned int lane_id;

    __device__ __forceinline__
    BlockScanWarpScans(TempStorage &temp_storage)
        : temp_storage(temp_storage.Alias()),
          linear_tid(threadIdx.x),
          warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS),
          lane_id(LaneId())
    {
    }

    static __device__ __forceinline__ void
    InclusiveScanShfl(int32_t input, int32_t &inclusive_output)
    {
        inclusive_output = input;

// Iterate scan steps
#pragma unroll
        for (unsigned int STEP = 0; STEP < STEPS; STEP++) {
            auto offset = 1u << STEP;
#ifdef NO_ASM_MAGIC
            auto word = __shfl_up_sync(MEMBER_MASK,
                                       inclusive_output,
                                       offset,
                                       WARP_THREADS);
            if (LaneId() >= offset) {
                inclusive_output += word;
            }
#else
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
#endif
        }
    }

    /// Use the warp-wide aggregates to compute the calling warp's prefix.  Also
    /// returns block-wide aggregate in all threads.
    __device__ __forceinline__ int32_t
    ComputeWarpPrefix(int32_t warp_aggregate, int32_t &block_aggregate)
    {
        // Last lane in each warp shares its warp-aggregate
        if (lane_id == WARP_THREADS - 1) {
            temp_storage.warp_aggregates[warp_id] = warp_aggregate;
        }

        __syncthreads();

        // Accumulate block aggregates and save the one that is our warp's
        // prefix
        int32_t warp_prefix;
        block_aggregate = temp_storage.warp_aggregates[0];

#pragma unroll
        for (auto WARP = 1u; WARP < WARPS; ++WARP) {
            if (warp_id == WARP)
                warp_prefix = block_aggregate;

            block_aggregate += temp_storage.warp_aggregates[WARP];
        }

        return warp_prefix;
    }

    //---------------------------------------------------------------------
    // Exclusive scans
    //---------------------------------------------------------------------

    /// Computes an exclusive thread block-wide prefix scan using the specified
    /// binary \p scan_op functor.  Each thread contributes one input element.
    /// Also provides every thread with the block-wide \p block_aggregate of all
    /// inputs. With no initial value, the output computed for
    /// <em>thread</em><sub>0</sub> is undefined.
    __device__ __forceinline__ void
    ExclusiveScan(
        const int32_t input,       ///< [in] Calling thread's input item
        int32_t &exclusive_output, ///< [out] Calling thread's output item (may
                                   ///< be aliased to \p input)
        int32_t &block_aggregate)  ///< [out] Threadblock-wide aggregate
                                   ///< reduction of input items
    {
        // Compute warp scan in each warp.  The exclusive output from each lane0
        // is invalid.
        int32_t inclusive_output;
        InclusiveScanShfl(input, inclusive_output);
        exclusive_output = inclusive_output - input;

        // Compute the warp-wide prefix and block-wide aggregate for each warp.
        int32_t warp_prefix =
            ComputeWarpPrefix(inclusive_output, block_aggregate);

        // Apply warp prefix to our lane's partial
        if (warp_id != 0) {
            exclusive_output += warp_prefix;
        }
    }

    /// Computes an exclusive thread block-wide prefix scan using the specified
    /// binary \p scan_op functor.  Each thread contributes one input element.
    /// the call-back functor \p block_prefix_callback_op is invoked by the
    /// first warp in the block, and the value returned by
    /// <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that
    /// logically prefixes the thread block's scan inputs.  Also provides every
    /// thread with the block-wide \p block_aggregate of all inputs.
    __device__ __forceinline__ void
    ExclusiveScan(
        const int32_t input,       ///< [in] Calling thread's input item
        int32_t &exclusive_output, ///< [out] Calling thread's output item (may
                                   ///< be aliased to \p input)
        TilePrefixCallbackOp
            &block_prefix_callback_op) ///< [in-out]
                                       ///< <b>[<em>warp</em><sub>0</sub>
                                       ///< only]</b> Call-back functor for
                                       ///< specifying a thread block-wide
                                       ///< prefix to be applied to all inputs.
    {
        // Compute block-wide exclusive scan.  The exclusive output from tid0 is
        // invalid.
        int32_t block_aggregate;
        ExclusiveScan(input, exclusive_output, block_aggregate);

        // Use the first warp to determine the thread block prefix, returning
        // the result in lane0
        if (warp_id == 0) {
            int32_t block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0) {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
            }
        }

        __syncthreads();

        // Incorporate thread block prefix into outputs
        int32_t block_prefix = temp_storage.block_prefix;
        exclusive_output += block_prefix;
    }

    //---------------------------------------------------------------------
    // Inclusive scans
    //---------------------------------------------------------------------

    /// Computes an inclusive thread block-wide prefix scan using the specified
    /// binary \p scan_op functor.  Each thread contributes one input element.
    /// Also provides every thread with the block-wide \p block_aggregate of all
    /// inputs.
    __device__ __forceinline__ void
    InclusiveScan(
        int32_t input,             ///< [in] Calling thread's input item
        int32_t &inclusive_output, ///< [out] Calling thread's output item (may
                                   ///< be aliased to \p input)
        int32_t &block_aggregate)  ///< [out] Threadblock-wide aggregate
                                   ///< reduction of input items
    {
        InclusiveScanShfl(input, inclusive_output);

        // Compute the warp-wide prefix and block-wide aggregate for each warp.
        // Warp prefix for warp0 is invalid.
        int32_t warp_prefix =
            ComputeWarpPrefix(inclusive_output, block_aggregate);

        // Apply warp prefix to our lane's partial
        if (warp_id != 0) {
            inclusive_output += warp_prefix;
        }
    }

    /// Computes an inclusive thread block-wide prefix scan using the specified
    /// binary \p scan_op functor.  Each thread contributes one input element.
    /// the call-back functor \p block_prefix_callback_op is invoked by the
    /// first warp in the block, and the value returned by
    /// <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that
    /// logically prefixes the thread block's scan inputs.  Also provides every
    /// thread with the block-wide \p block_aggregate of all inputs.
    __device__ __forceinline__ void
    InclusiveScan(
        int32_t input,             ///< [in] Calling thread's input item
        int32_t &exclusive_output, ///< [out] Calling thread's output item (may
                                   ///< be aliased to \p input)
        TilePrefixCallbackOp
            &block_prefix_callback_op) ///< [in-out]
                                       ///< <b>[<em>warp</em><sub>0</sub>
                                       ///< only]</b> Call-back functor for
                                       ///< specifying a thread block-wide
                                       ///< prefix to be applied to all inputs.
    {
        int32_t block_aggregate;
        InclusiveScan(input, exclusive_output, block_aggregate);

        // Use the first warp to determine the thread block prefix, returning
        // the result in lane0
        if (warp_id == 0) {
            int32_t block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0) {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
            }
        }

        __syncthreads();

        // Incorporate thread block prefix into outputs
        int32_t block_prefix = temp_storage.block_prefix;
        exclusive_output += block_prefix;
    }
};

class BlockScan {
private:
    /// Shared memory storage layout type for BlockScan
    using _TempStorage = BlockScanWarpScans::TempStorage;

    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;

    /**
     * \brief Perform a sequential inclusive prefix scan over the
     * statically-sized \p input array, seeded with the specified \p prefix. The
     * aggregate is returned.
     *
     * \tparam LENGTH     <b>[inferred]</b> LengthT of \p input and \p output
     * arrays
     */
    template <int LENGTH>
    __device__ __forceinline__ int32_t
    ThreadScanInclusive(
        const int32_t (&input)[LENGTH], ///< [in] Input array
        int32_t (&output)[LENGTH], ///< [out] Output array (may be aliased to \p
                                   ///< input)
        int32_t prefix,            ///< [in] Prefix to seed scan with
        bool apply_prefix =
            true) ///< [in] Whether or not the calling thread should apply its
                  ///< prefix. (Handy for preventing thread-0 from applying a
                  ///< prefix.)
    {
        int32_t inclusive = input[0];
        if (apply_prefix) {
            inclusive += prefix;
        }
        output[0] = inclusive;

#pragma unroll
        for (int i = 1; i < LENGTH; ++i) {
            inclusive += input[i];
            output[i] = inclusive;
        }

        return inclusive;
    }

    template <int LENGTH>
    __device__ __forceinline__ int32_t
    ThreadReduce(const int32_t *const input, ///< [in] Input array
                 int32_t prefix, ///< [in] Prefix to seed reduction with
                 Int2Type<LENGTH> /*length*/)
    {
        int32_t retval = prefix;

#pragma unroll
        for (int i = 0; i < LENGTH; ++i)
            retval += input[i];

        return retval;
    }

    /**
     * \brief Perform a sequential reduction over \p LENGTH elements of the \p
     * input array, seeded with the specified \p prefix.  The aggregate is
     * returned.
     *
     * \tparam LENGTH     LengthT of input array
     */
    template <int LENGTH>
    __device__ __forceinline__ int32_t
    ThreadReduce(const int32_t *const input, ///< [in] Input array
                 int32_t prefix) ///< [in] Prefix to seed reduction with
    {
        return ThreadReduce(input, prefix, Int2Type<LENGTH>());
    }

    /**
     * \brief Perform a sequential reduction over \p LENGTH elements of the \p
     * input array.  The aggregate is returned.
     *
     * \tparam LENGTH     LengthT of input array
     */
    template <int LENGTH>
    __device__ __forceinline__ int32_t
    ThreadReduce(const int32_t *const input) ///< [in] Input array
    {
        int32_t prefix = input[0];
        return ThreadReduce<LENGTH - 1>(input + 1, prefix);
    }

    /**
     * \brief Perform a sequential reduction over the statically-sized \p input
     * array, seeded with the specified \p prefix. The aggregate is returned.
     *
     * \tparam LENGTH     <b>[inferred]</b> LengthT of \p input array
     */
    template <int LENGTH>
    __device__ __forceinline__ int32_t
    ThreadReduce(int32_t (&input)[LENGTH], ///< [in] Input array
                 int32_t prefix) ///< [in] Prefix to seed reduction with
    {
        return ThreadReduce(input, prefix, Int2Type<LENGTH>());
    }

    /**
     * \brief Serial reduction with the specified operator
     *
     * \tparam LENGTH     <b>[inferred]</b> LengthT of \p input array
     */
    template <int LENGTH>
    __device__ __forceinline__ int32_t
    ThreadReduce(const int32_t (&input)[LENGTH]) ///< [in] Input array
    {
        return ThreadReduce<LENGTH>(static_cast<const int32_t *>(input));
    }

    /******************************************************************************
     * Public types
     ******************************************************************************/
public:
    /// \smemstorage{BlockScan}
    using TempStorage = Uninitialized<_TempStorage>;

    /**
     * \brief Collective constructor using the specified memory allocation as
     * temporary storage.
     */
    __device__ __forceinline__
    BlockScan(
        TempStorage &temp_storage) ///< [in] Reference to memory allocation
                                   ///< having layout type TempStorage
        : temp_storage(temp_storage.Alias()),
          linear_tid(threadIdx.x)
    {
    }

    __device__ __forceinline__ void
    ExclusiveSum(int32_t input,   ///< [in] Calling thread's input item
                 int32_t &output) ///< [out] Calling thread's output item (may
                                  ///< be aliased to \p input)
    {
        int32_t initial_value{};

        BlockScanWarpScans(temp_storage)
            .ExclusiveScan(input, output, initial_value);
    }

    template <typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void
    ExclusiveSum(
        int32_t input,   ///< [in] Calling thread's input item
        int32_t &output, ///< [out] Calling thread's output item (may
                         ///< be aliased to \p input)
        BlockPrefixCallbackOp
            &block_prefix_callback_op) ///< [in-out]
                                       ///< <b>[<em>warp</em><sub>0</sub>
                                       ///< only]</b> Call-back functor
                                       ///< for specifying a block-wide
                                       ///< prefix to be applied to the
                                       ///< logical input sequence.
    {
        ExclusiveScan(input, output, block_prefix_callback_op);
    }

    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified
     * binary \p scan_op functor.  Each thread contributes one input element.
     * Also provides every thread with the block-wide \p block_aggregate of all
     * inputs. With no initial value, the output computed for
     * <em>thread</em><sub>0</sub> is undefined.
     */
    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void
    ExclusiveScan(
        int32_t (
            &input)[ITEMS_PER_THREAD], ///< [in] Calling thread's input items
        int32_t (
            &output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                        ///< (may be aliased to \p input)
        int32_t &block_aggregate) ///< [out] block-wide aggregate reduction of
                                  ///< input items
    {
        // Reduce consecutive thread items in registers
        int32_t thread_partial = ThreadReduce(input);

        // Exclusive thread block-scan
        BlockScanWarpScans(temp_storage)
            .ExclusiveScan(thread_partial, thread_partial, block_aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, thread_partial, (linear_tid != 0));
    }

    template <typename CallBackOrAgg>
    __device__ __forceinline__ void
    InclusiveScan(
        int32_t input,   ///< [in] Calling thread's input item
        int32_t &output, ///< [out] Calling thread's output item (may
                         ///< be aliased to \p input)
        CallBackOrAgg
            &block_prefix_callback_op) ///< [in-out]
                                       ///< <b>[<em>warp</em><sub>0</sub>
                                       ///< only]</b> Call-back functor
                                       ///< for specifying a block-wide
                                       ///< prefix to be applied to the
                                       ///< logical input sequence.
    {
        BlockScanWarpScans(temp_storage)
            .InclusiveScan(input, output, block_prefix_callback_op);
    }

    __device__ __forceinline__ void
    InclusiveSum(const int32_t input, ///< [in] Calling thread's input item
                 int32_t &output, ///< [out] Calling thread's output item (may
                                  ///< be aliased to \p input)
                 int32_t &block_aggregate) ///< [out] block-wide aggregate
                                           ///< reduction of input items
    {
        InclusiveScan(input, output, block_aggregate);
    }

    __device__ __forceinline__ void
    InclusiveSum(
        int32_t input,   ///< [in] Calling thread's input item
        int32_t &output, ///< [out] Calling thread's output item (may
                         ///< be aliased to \p input)
        TilePrefixCallbackOp
            &block_prefix_callback_op) ///< [in-out]
                                       ///< <b>[<em>warp</em><sub>0</sub>
                                       ///< only]</b> Call-back functor
                                       ///< for specifying a block-wide
                                       ///< prefix to be applied to the
                                       ///< logical input sequence.
    {
        InclusiveScan(input, output, block_prefix_callback_op);
    }

    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void
    InclusiveSum(
        int32_t (
            &input)[ITEMS_PER_THREAD], ///< [in] Calling thread's input items
        int32_t (
            &output)[ITEMS_PER_THREAD]) ///< [out] Calling thread's output items
                                        ///< (may be aliased to \p input)
    {
        if constexpr (ITEMS_PER_THREAD == 1) {
            InclusiveSum(input[0], output[0]);
        } else {
            // Reduce consecutive thread items in registers
            int32_t thread_prefix = ThreadReduce(input);

            // Exclusive thread block-scan
            ExclusiveSum(thread_prefix, thread_prefix);

            // Inclusive scan in registers with prefix as seed
            ThreadScanInclusive(input,
                                output,
                                thread_prefix,
                                (linear_tid != 0));
        }
    }

    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void
    InclusiveSum(
        int32_t (
            &input)[ITEMS_PER_THREAD], ///< [in] Calling thread's input items
        int32_t (
            &output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                        ///< (may be aliased to \p input)
        int32_t &block_aggregate) ///< [out] block-wide aggregate reduction of
                                  ///< input items
    {
        if constexpr (ITEMS_PER_THREAD == 1) {
            InclusiveSum(input[0], output[0], block_aggregate);
        } else {
            // Reduce consecutive thread items in registers
            int32_t thread_prefix = ThreadReduce(input);

            // Exclusive thread block-scan
            ExclusiveSum(thread_prefix, thread_prefix, block_aggregate);

            // Inclusive scan in registers with prefix as seed
            ThreadScanInclusive(input,
                                output,
                                thread_prefix,
                                (linear_tid != 0));
        }
    }

    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ void
    InclusiveSum(
        int32_t (
            &input)[ITEMS_PER_THREAD], ///< [in] Calling thread's input items
        int32_t (
            &output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                        ///< (may be aliased to \p input)
        TilePrefixCallbackOp
            &block_prefix_callback_op) ///< [in-out]
                                       ///< <b>[<em>warp</em><sub>0</sub>
                                       ///< only]</b> Call-back functor for
                                       ///< specifying a block-wide prefix to be
                                       ///< applied to the logical input
                                       ///< sequence.
    {
        if constexpr (ITEMS_PER_THREAD == 1) {
            InclusiveSum(input[0], output[0], block_prefix_callback_op);
        } else {
            // Reduce consecutive thread items in registers
            int32_t thread_prefix = ThreadReduce(input);

            // Exclusive thread block-scan
            ExclusiveSum(thread_prefix,
                         thread_prefix,
                         block_prefix_callback_op);

            // Inclusive scan in registers with prefix as seed
            ThreadScanInclusive(input, output, thread_prefix);
        }
    }

    template <int ITEMS_PER_THREAD,
              typename CallbackOrAgg>
    __device__ __forceinline__ void
    InclusiveScan(
        const int32_t (
            &input)[ITEMS_PER_THREAD], ///< [in] Calling thread's input items
        int32_t (
            &output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                        ///< (may be aliased to \p input)
        CallbackOrAgg
            &block_prefix_callback_op_or_agg) ///< [in-out]
                                              ///< <b>[<em>warp</em><sub>0</sub>
                                              ///< only]</b> Call-back functor
                                              ///< for specifying a block-wide
                                              ///< prefix to be applied to the
                                              ///< logical input sequence.
    {
        if constexpr (ITEMS_PER_THREAD == 1) {
            InclusiveScan(input[0], output[0], block_prefix_callback_op_or_agg);
        } else {
            // Reduce consecutive thread items in registers
            int32_t thread_prefix = ThreadReduce(input);

            // Exclusive thread block-scan
            BlockScanWarpScans(temp_storage)
                .ExclusiveScan(thread_prefix,
                               thread_prefix,
                               block_prefix_callback_op_or_agg);

            // Inclusive scan in registers with prefix as seed
            ThreadScanInclusive(input, output, thread_prefix);
        }
    }
};

struct AgentScan {
    static constexpr unsigned int TILE_ITEMS =
        ScanPolicy::BLOCK_THREADS * ScanPolicy::THREAD_ITEMS;

    union _TempStorage {
        // Smem needed for tile loading
        BlockLoad::TempStorage load;

        // Smem needed for tile storing
        BlockStore::TempStorage store;

        struct ScanStorage {
            // Smem needed for cooperative prefix callback
            TilePrefixCallbackOp::TempStorage prefix;

            // Smem needed for tile scanning
            BlockScan::TempStorage scan;
        } scan_storage;
    };

    using TempStorage = Uninitialized<_TempStorage>;

    _TempStorage &temp_storage; ///< Reference to temp_storage
    const int32_t *d_in;        ///< Input data
    int32_t *d_out;             ///< Output data

    __device__ __forceinline__
    AgentScan(TempStorage &temp_storage, const int32_t *d_in, int32_t *d_out)
        : temp_storage(temp_storage.Alias()),
          d_in(d_in),
          d_out(d_out)
    {
    }

    template <bool IS_LAST_TILE>
    __device__ __forceinline__ void
    ConsumeTile(size_t num_remaining,
                const unsigned int tile_idx,
                const unsigned int tile_offset,
                ScanTileState &tile_state)
    {
        // Load items
        int32_t items[ScanPolicy::THREAD_ITEMS];

        if constexpr (IS_LAST_TILE) {
            // Fill last element with the first element because collectives are
            // not suffix guarded.
            BlockLoad(temp_storage.load)
                .Load(d_in + tile_offset,
                      items,
                      num_remaining,
                      *(d_in + tile_offset));
        } else {
            BlockLoad(temp_storage.load).Load(d_in + tile_offset, items);
        }

        __syncthreads();

        // Perform tile scan
        BlockScan tile_scanner(temp_storage.scan_storage.scan);
        if (tile_idx == 0) {
            // Scan first tile
            int32_t block_aggregate;
            tile_scanner.InclusiveScan(items, items, block_aggregate);

            if constexpr (!IS_LAST_TILE) {
                if (threadIdx.x == 0) {
                    tile_state.SetInclusive(0, block_aggregate);
                }
            }
        } else {
            // Scan non-first tile
            TilePrefixCallbackOp prefix_op(tile_state,
                                           temp_storage.scan_storage.prefix,
                                           tile_idx);
            tile_scanner.InclusiveScan(items, items, prefix_op);
        }

        __syncthreads();

        // Store items
        if constexpr (IS_LAST_TILE) {
            BlockStore(temp_storage.store)
                .Store(d_out + tile_offset, items, num_remaining);
        } else {
            BlockStore(temp_storage.store).Store(d_out + tile_offset, items);
        }
    }

    __device__ __forceinline__ void
    ConsumeRange(size_t num_items,
                 ScanTileState &tile_state,
                 unsigned int start_tile)
    {
        // Blocks are launched in increasing order, so just assign one tile per
        // block

        // Current tile index
        auto tile_idx = start_tile + blockIdx.x;

        // Global offset (in terms of items) for the current tile
        auto tile_offset = TILE_ITEMS * tile_idx;

        // Remaining items (including this tile), shouldn't underflow since
        // there should be no extra tile
        auto num_remaining = num_items - tile_offset;

        if (num_remaining > TILE_ITEMS) {
            // Not last tile
            ConsumeTile<false>(num_remaining,
                               tile_idx,
                               tile_offset,
                               tile_state);
        } else if (num_remaining > 0) {
            // Last tile
            ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
        }
    }
};

template <int ALLOCATIONS = 1>
__host__ __forceinline__ cudaError_t
AliasTemporaries(
    void *d_temp_storage, ///< [in] Device-accessible allocation of temporary
                          ///< storage.  When NULL, the required allocation size
                          ///< is written to \p temp_storage_bytes and no work
                          ///< is done.
    size_t &temp_storage_bytes, ///< [in,out] Size in bytes of \t d_temp_storage
                                ///< allocation
    void *&allocations, ///< [in,out] Pointers to device allocations needed
    size_t
        allocation_sizes) ///< [in] Sizes in bytes of device allocations needed
{
    constexpr size_t ALIGN_BYTES = 256;
    constexpr auto ALIGN_MASK = ~(ALIGN_BYTES - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t allocation_offsets;
    size_t bytes_needed = 0;
    size_t allocation_bytes = (allocation_sizes + ALIGN_BYTES - 1) & ALIGN_MASK;
    allocation_offsets = bytes_needed;
    bytes_needed += allocation_bytes;
    bytes_needed += ALIGN_BYTES - 1;

    // Check if the caller is simply requesting the size of the storage
    // allocation
    if (!d_temp_storage) {
        temp_storage_bytes = bytes_needed;
        return cudaSuccess;
    }

    // Check if enough storage provided
    if (temp_storage_bytes < bytes_needed) {
        return check_cuda_error(cudaErrorInvalidValue);
    }

    // Alias
    d_temp_storage = (void *)((size_t(d_temp_storage) + ALIGN_BYTES - 1) &
                              static_cast<size_t>(ALIGN_MASK));
    allocations = static_cast<char *>(d_temp_storage) + allocation_offsets;

    return cudaSuccess;
}

__global__ void
DeviceScanInitKernel(ScanTileState tile_state, size_t num_tiles)
{
    tile_state.InitializeStatus(num_tiles);
}

__launch_bounds__(ScanPolicy::BLOCK_THREADS) __global__
    void DeviceScanKernel(const int32_t *d_in,
                          int32_t *d_out,
                          ScanTileState tile_state,
                          unsigned int start_tile,
                          size_t num_items)
{
    // Shared memory for AgentScan
    __shared__ AgentScan::TempStorage temp_storage;

    // Process tiles
    AgentScan(temp_storage, d_in, d_out)
        .ConsumeRange(num_items, tile_state, start_tile);
}

__host__ cudaError_t
InclusiveSum(void *d_temp_storage,
             size_t &temp_storage_bytes,
             const int32_t *d_in,
             int32_t *d_out,
             size_t num_items,
             cudaStream_t stream = 0)
{
    static constexpr auto INIT_KERNEL_THREADS = 128;

    constexpr auto ret = cudaSuccess;

    // Get device ordinal
    int device_ordinal;
    check_cuda_error(cudaGetDevice(&device_ordinal));

    // Number of input tiles
    constexpr auto tile_size =
        ScanPolicy::BLOCK_THREADS * ScanPolicy::THREAD_ITEMS;
    const auto num_tiles = (num_items + tile_size - 1) / tile_size;

    // Specify temporary storage allocation requirements
    size_t allocation_sizes = ScanTileState::AllocationSize(num_tiles);

    // Compute allocation pointers into the single storage blob (or compute the
    // necessary size of the blob)
    void *allocations = nullptr;
    check_cuda_error(AliasTemporaries(d_temp_storage,
                                      temp_storage_bytes,
                                      allocations,
                                      allocation_sizes));
    if (!d_temp_storage) {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        return ret;
    }

    // Return if empty problem
    if (num_items == 0)
        return ret;

    // Construct the tile status interface
    ScanTileState tile_state;
    check_cuda_error(tile_state.Init(allocations));

    // Log init_kernel configuration
    auto init_grid_size = static_cast<unsigned int>(
        (num_tiles + INIT_KERNEL_THREADS - 1) / INIT_KERNEL_THREADS);

    // Invoke init_kernel to initialize tile descriptors
    DeviceScanInitKernel<<<init_grid_size, INIT_KERNEL_THREADS, 0, stream>>>(
        tile_state,
        num_tiles);

    // Check for failure to launch
    check_cuda_error(cudaPeekAtLastError());

    // Get max x-dimension of grid
    int max_dim_x;
    check_cuda_error(cudaDeviceGetAttribute(&max_dim_x,
                                            cudaDevAttrMaxGridDimX,
                                            device_ordinal));

    // Run grids in epochs (in case number of tiles exceeds max x-dimension
    auto scan_grid_size =
        static_cast<unsigned int>(mixedMin(num_tiles, max_dim_x));
    for (unsigned int start_tile = 0; start_tile < num_tiles;
         start_tile += scan_grid_size) {
        // Invoke scan_kernel
        DeviceScanKernel<<<scan_grid_size,
                           ScanPolicy::BLOCK_THREADS,
                           0,
                           stream>>>(d_in,
                                     d_out,
                                     tile_state,
                                     start_tile,
                                     num_items);

        // Check for failure to launch
        check_cuda_error(cudaPeekAtLastError());
    }

    return cudaSuccess;
}

void
impl_cub_simplified(const int32_t *d_input, int32_t *d_output, size_t size)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, size);
    cudaFree(d_temp_storage);
}
