
void
impl_serial_cpu_baseline(const int32_t *h_input, int32_t *h_output, size_t size)
{
    int32_t ans = 0;
    for (size_t i = 0; i < size; ++i) {
        ans += h_input[i];
        h_output[i] = ans;
    }
}