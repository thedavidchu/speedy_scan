#include <cub/cub.cuh>

void
impl_nvidia_decoupled_lookback(const int32_t *d_input,
                               int32_t *d_output,
                               size_t size)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    auto num_items = static_cast<int>(size);
    cub::DeviceScan::InclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  d_input,
                                  d_output,
                                  num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  d_input,
                                  d_output,
                                  num_items);
    cudaFree(d_temp_storage);
    cudaDeviceSynchronize();
}
