
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
/// SERIAL CPU IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

void
impl_serial_cpu(const int32_t *h_input, int32_t *h_output, size_t size)
{
    int32_t ans = 0;
    for (size_t i = 0; i < size; ++i) {
        ans += h_input[i];
        h_output[i] = ans;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// PARALLEL CPU IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

static void
partion_scan(int32_t const *const h_input,
             int32_t *const h_output,
             const size_t size,
             const unsigned worker_id,
             const unsigned num_workers)
{
    const size_t start_id = worker_id * size / num_workers;
    const size_t end_id = (worker_id + 1) * size / num_workers; // One past the end
    const size_t work_length = end_id - start_id;

    // Add inside thread's partition
    impl_serial_cpu(&h_input[start_id],
                             &h_output[start_id],
                             work_length);
}

static void
propagate_reduction(int32_t *const h_output,
                    int32_t *const reductions,
                    const size_t size,
                    const unsigned worker_id,
                    const unsigned num_workers)
{
    int32_t accum = 0;
    // Add the last element of each worker up-to-and-including your own!
    for (unsigned i = 0; i <= worker_id; ++i) {
        const size_t last_elem_id = (i + 1) * size / num_workers - 1;
        accum += h_output[last_elem_id];
    }
    reductions[worker_id] = accum;
}

static void
add_reduction(int32_t *const h_output,
              int32_t *const reductions,
              const size_t size,
              const unsigned worker_id,
              const unsigned num_workers)
{
    if (worker_id == 0) {
        return;
    }

    const int32_t base = reductions[worker_id - 1];
    const size_t start_id = worker_id * size / num_workers;
    const size_t end_id = (worker_id + 1) * size / num_workers; // One past the end

    for (size_t i = start_id; i < end_id; ++i) {
        h_output[i] += base;
    }
}

void
impl_parallel_cpu(const int32_t *h_input,
                           int32_t *h_output,
                           size_t size,
                           unsigned num_workers)
{
    assert(h_input != NULL && "input must not be NULL");
    assert(h_output != NULL && "output must not be NULL");
    assert(num_workers > 0 && "must have non-zero workers");

    // Scan the partitions
    std::vector<std::thread> workers;
    for (unsigned w_id = 0; w_id < num_workers; ++w_id) {
        workers.emplace_back(partion_scan,
                             h_input,
                             h_output,
                             size,
                             w_id,
                             num_workers);
    }
    for (auto &worker : workers) {
        worker.join();
    }

    // Propagate reductions
    std::vector<int32_t> reductions(num_workers);
    workers.clear();
    for (unsigned w_id = 0; w_id < num_workers; ++w_id) {
        workers.emplace_back(propagate_reduction,
                             h_output,
                             reductions.data(),
                             size,
                             w_id,
                             num_workers);
    }
    for (auto &worker : workers) {
        worker.join();
    }

    // Add reductions
    workers.clear();
    for (unsigned w_id = 0; w_id < num_workers; ++w_id) {
        workers.emplace_back(add_reduction,
                             h_output,
                             reductions.data(),
                             size,
                             w_id,
                             num_workers);
    }
    for (auto &worker : workers) {
        worker.join();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// OPTIMAL PARALLEL CPU IMPLEMENTATION -- BUT INCORRECT (it's just a memcpy)
////////////////////////////////////////////////////////////////////////////////

static void
optimal_but_incorrect_worker(const int32_t *h_input,
                             int32_t *h_output,
                             size_t size,
                             unsigned worker_id,
                             unsigned num_workers)
{
    const size_t start_id = worker_id * size / num_workers;
    const size_t end_id = (worker_id + 1) * size / num_workers;
    const size_t work_length = end_id - start_id;
    memcpy(&h_output[start_id],
           &h_input[start_id],
           work_length * sizeof(int32_t));
}

void
impl_simulate_optimal_but_incorrect_cpu(const int32_t *h_input,
                                        int32_t *h_output,
                                        size_t size,
                                        unsigned num_workers)
{
    assert(h_input != NULL && "input must not be NULL");
    assert(h_output != NULL && "output must not be NULL");
    assert(num_workers > 0 && "must have non-zero workers");

    std::vector<std::thread> workers;
    for (unsigned w_id = 0; w_id < num_workers; ++w_id) {
        workers.emplace_back(optimal_but_incorrect_worker,
                             h_input,
                             h_output,
                             size,
                             w_id,
                             num_workers);
    }

    for (auto &worker : workers) {
        worker.join();
    }
}
