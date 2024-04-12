#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <stdio.h>
#include <sys/time.h>

#include "common.hpp"
#include "generate_input.hpp"

// defined in each of the implementations
extern void
impl_serial_cpu(const int32_t *input, int32_t *output, size_t size);

extern void
impl_std_serial_cpu(const int32_t *input, int32_t *output, size_t size);

extern void
impl_parallel_cpu(const int32_t *h_input,
                  int32_t *h_output,
                  size_t size,
                  unsigned num_workers);

extern void
impl_simulate_optimal_but_incorrect_cpu(const int32_t *h_input,
                                        int32_t *h_output,
                                        size_t size,
                                        unsigned num_workers);

extern void
impl_serial_gpu(const int32_t *d_input, int32_t *d_output, size_t size);

void
impl_naive_hierarchical_gpu(const int32_t *d_input,
                            int32_t *d_output,
                            size_t size);

extern void
impl_optimized_hierarchical_gpu(const int32_t *input,
                                int32_t *output,
                                size_t size);

extern void
impl_our_decoupled_lookback(const int32_t *input, int32_t *output, size_t size);

extern void
impl_nvidia_decoupled_lookback(const int32_t *input,
                               int32_t *output,
                               size_t size);

extern void
impl_simulate_optimal_but_incorrect_gpu(const int32_t *d_input,
                                        int32_t *d_output,
                                        size_t size);

extern void
impl_cub_simplified(const int32_t *d_input, int32_t *d_output, size_t size);

enum class InclusiveScanType {
    CPU_Serial,
    CPU_StdSerial,
    CPU_Parallel,
    CPU_SimulateOptimalButIncorrect,
    GPU_Serial,
    GPU_NaiveHierarchical,
    GPU_OptimizedHierarchical,
    GPU_OurDecoupledLookback,
    GPU_NvidiaDecoupledLookback,
    GPU_SimulateOptimalButIncorrect,
    GPU_CUBSimplified,
};

// I just use this as an associative array
std::vector<std::pair<std::string, InclusiveScanType>> scan_types = {
    // CPU Algorithms
    {"CPU_Serial", InclusiveScanType::CPU_Serial},
    {"CPU_StdSerial", InclusiveScanType::CPU_StdSerial},
    {"CPU_Parallel", InclusiveScanType::CPU_Parallel},
    {"CPU_SimulateOptimalButIncorrect",
     InclusiveScanType::CPU_SimulateOptimalButIncorrect},
    // GPU Algorithms
    {"GPU_Serial", InclusiveScanType::GPU_Serial},
    {"GPU_NaiveHierarchical", InclusiveScanType::GPU_NaiveHierarchical},
    {"GPU_OptimizedHierarchical", InclusiveScanType::GPU_OptimizedHierarchical},
    {"GPU_OurDecoupledLookback", InclusiveScanType::GPU_OurDecoupledLookback},
    {"GPU_NvidiaDecoupledLookback",
     InclusiveScanType::GPU_NvidiaDecoupledLookback},
    {"GPU_SimulateOptimalButIncorrect",
     InclusiveScanType::GPU_SimulateOptimalButIncorrect},
    {"GPU_CUBSimplified", InclusiveScanType::GPU_CUBSimplified},
};

static bool
is_gpu_algorithm(InclusiveScanType scan_type)
{
    switch (scan_type) {
    case InclusiveScanType::GPU_Serial:
    case InclusiveScanType::GPU_NaiveHierarchical:
    case InclusiveScanType::GPU_OptimizedHierarchical:
    case InclusiveScanType::GPU_OurDecoupledLookback:
    case InclusiveScanType::GPU_NvidiaDecoupledLookback:
    case InclusiveScanType::GPU_SimulateOptimalButIncorrect:
    case InclusiveScanType::GPU_CUBSimplified:
        return true;
    default:
        return false;
    }
}

static void
print_warning(std::string msg)
{
    std::cerr << "Warning: " << msg << std::endl;
}

static void
print_error(std::string msg)
{
    std::cerr << "Error: " << msg << std::endl;
}

struct CommandLineArguments {
    std::string exe_ = "?";
    // TODO(dchu)   Make these values into macros or somehow deduplicate the
    //              references to them!
    InclusiveScanType type_ = InclusiveScanType::GPU_OptimizedHierarchical;
    int size_ = 1000;
    int repeats_ = 1;
    bool check_ = false;
    bool debug_ = false;

    CommandLineArguments(int argc, char *argv[]);

    void
    print();

private:
    static bool
    matches_str(char *arg, std::vector<std::string> flags)
    {
        for (auto flag : flags) {
            if (strcmp(arg, flag.c_str()) == 0) {
                return true;
            }
        }
        return false;
    }

    void
    print_help();

    InclusiveScanType
    parse_inclusive_scan_type(char *arg);

    int
    parse_positive_int(char *arg);

    std::string
    inclusive_scan_type_to_input_string(InclusiveScanType type);

    static inline std::string
    bool_to_string(bool x)
    {
        return x ? "true" : "false";
    }
};

CommandLineArguments::CommandLineArguments(int argc, char *argv[])
{
    this->exe_ = argv[0];

    // Set parameters based on user arguments
    for (int i = 1; i < argc; ++i) {
        if (matches_str(argv[i], {"-t", "--type"})) {
            ++i;
            if (i >= argc) {
                print_error("expecting scan type!");
                print_help();
                exit(-1);
            }
            this->type_ = parse_inclusive_scan_type(argv[i]);
        } else if (matches_str(argv[i], {"--size", "-s"})) {
            ++i;
            if (i >= argc) {
                print_error("expecting size!");
                print_help();
                exit(-1);
            }
            this->size_ = parse_positive_int(argv[i]);
        } else if (matches_str(argv[i], {"--repeats", "-r"})) {
            ++i;
            if (i >= argc) {
                print_error("expecting repeats!");
                print_help();
                exit(-1);
            }
            this->repeats_ = parse_positive_int(argv[i]);
        } else if (matches_str(argv[i], {"--check", "-c"})) {
            this->check_ = true;
        } else if (matches_str(argv[i], {"--debug", "-d"})) {
            this->debug_ = true;
        } else if (matches_str(argv[i], {"--help", "-h"})) {
            print_help();
            exit(0);
        } else {
            // TODO(dchu): print out the unexpected argument that we received
            print_error("unexpected argument: '" + std::string(argv[i]) + "'!");
            print_help();
            exit(-1);
        }
    }
}

void
CommandLineArguments::print()
{
    std::cout << "CommandLineArguments(type='"
              << inclusive_scan_type_to_input_string(this->type_)
              << "', size=" << this->size_ << ", repeats=" << this->repeats_
              << ", check=" << bool_to_string(this->check_)
              << ", debug=" << bool_to_string(this->debug_) << ")" << std::endl;
}

void
CommandLineArguments::print_help()
{
    std::cout
        << "Usage: " << this->exe_
        << " [-t <scan-type>] [-s <input-size>] [-r <repetitions>] [-c] [-d]"
        << std::endl;
    std::cout << "    -t, --type <scan-type>: scan type, "
                 "{";
    for (auto [str, type] : scan_types) {
        std::cout << str << ",";
    }
    // Delete the trailing comma from the previous print-statement
    std::cout << "\b}. Default: GPU_OptimizedHierarchical" << std::endl;
    std::cout << "    -s, --size <input-size>: number of input elements, 1..= "
                 "~1_000_000_000. Default: 1000"
              << std::endl;
    std::cout << "    -r, --repeats <repetitions>: number of times the test is "
                 "repeated, 0..=MAX_UNSIGNED_INT. Default: 1"
              << std::endl;
    std::cout << "    -c, --check: check the output for correctness. "
                 "Default: off"
              << std::endl;
    std::cout << "    -d, --debug: debug mode, so populate the input with all "
                 "1's. Default: off"
              << std::endl;
    std::cout << "    -h, --help: print this help message. Overrides all else!"
              << std::endl;
}

InclusiveScanType
CommandLineArguments::parse_inclusive_scan_type(char *arg)
{
    // Yes, I know that I could just do the lookup, but this was simpler
    // than dealing with exceptions.
    for (auto [str, type] : scan_types) {
        if (matches_str(arg, {str})) {
            return type;
        }
    }
    print_error("unrecognized scan type: '" + std::string(arg) + "'!");
    print_help();
    exit(-1);
}

int
CommandLineArguments::parse_positive_int(char *arg)
{
    long x = 0;
    // NOTE This function strips leading whitespace and interprets until
    //      it reaches an invalid value. This can create odd results,
    //      such as '1e12' being interpreted as '1'.
    x = strtol(arg, NULL, 10);
    if (x > INT_MAX) {
        print_error("'" + std::string(arg) + "' is out of range (max int is " +
                    std::to_string(INT_MAX) + ")!");
        print_help();
        exit(-1);
    } else if (x <= 0) { // Unparseable non-integers will trigger this
        print_error("got '" + std::string(arg) +
                    "', expecting positive, integer value!");
        print_help();
        exit(-1);
    }
    return static_cast<int>(x);
}

std::string
CommandLineArguments::inclusive_scan_type_to_input_string(
    InclusiveScanType type)
{
    // Yes, I know that I could just do the lookup, but this was simpler
    // than dealing with exceptions.
    for (auto [str, the_type] : scan_types) {
        if (the_type == type) {
            return str;
        }
    }
    print_error("unrecognized scan type!");
    exit(-1);
}

double
get_time_in_seconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

void
print_duration(double start_time, double end_time)
{
    // NOTE I'm using the C printf function so that I can specify the format of
    //      the double's output string! We print 6 significant figures because
    //      we only measure to microsecond accuracy, which is 6 significant
    //      figures. Wow, I didn't realize this until now!
    printf("@@@ Elapsed time (sec): %.6f\n", end_time - start_time);
}

int
main(int argc, char *argv[])
{
    cudaDeviceReset();
    CommandLineArguments cmd_args(argc, argv);
    cmd_args.print();

    const size_t num_elems = cmd_args.size_;
    const size_t array_size_in_bytes = sizeof(int32_t) * num_elems;

    // Generate inputs
    std::vector<int32_t> h_input;
    if (cmd_args.debug_) {
        h_input = std::vector<int32_t>(cmd_args.size_, 1);
    } else {
        h_input = generate_input(cmd_args.size_, 0);
    }

    // Malloc CPU outputs
    int32_t *h_output = nullptr;
    if (!is_gpu_algorithm(cmd_args.type_)) {
        h_output = (int32_t *)malloc(array_size_in_bytes);
        assert(h_output && "oom");
    }

    // Allocate memory
    int32_t *d_input = nullptr;
    int32_t *d_output = nullptr;
    if (is_gpu_algorithm(cmd_args.type_)) {
        cuda_check(cudaMalloc((void **)&d_input, array_size_in_bytes),
                   "cudaMalloc(d_input)");
        cuda_check(cudaMalloc((void **)&d_output, array_size_in_bytes),
                   "cudaMalloc(d_output)");
    }

    // Copy input from host to device
    if (is_gpu_algorithm(cmd_args.type_)) {
        cuda_check(cudaMemcpy(d_input,
                              h_input.data(),
                              array_size_in_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(H2D)");
    }

    for (int i = 0; i < cmd_args.repeats_; ++i) {
        const double start_time = get_time_in_seconds();
        switch (cmd_args.type_) {
        case InclusiveScanType::CPU_Serial:
            impl_serial_cpu(h_input.data(), h_output, num_elems);
            break;
        case InclusiveScanType::CPU_StdSerial:
            impl_std_serial_cpu(h_input.data(), h_output, num_elems);
            break;
        case InclusiveScanType::CPU_Parallel:
            impl_parallel_cpu(h_input.data(), h_output, num_elems, 16);
            break;
        case InclusiveScanType::CPU_SimulateOptimalButIncorrect:
            if (cmd_args.check_) {
                print_warning("CPU_SimulateOptimalButIncorrect does not return "
                              "the correct answer; it merely simulates the "
                              "optimal timing with a memcpy!");
            }
            impl_simulate_optimal_but_incorrect_cpu(h_input.data(),
                                                    h_output,
                                                    num_elems,
                                                    16);
            break;
        case InclusiveScanType::GPU_Serial:
            impl_serial_gpu(d_input, d_output, num_elems);
            break;
        case InclusiveScanType::GPU_NaiveHierarchical:
            impl_naive_hierarchical_gpu(d_input, d_output, num_elems);
            break;
        case InclusiveScanType::GPU_OptimizedHierarchical:
            impl_optimized_hierarchical_gpu(d_input, d_output, num_elems);
            break;

        case InclusiveScanType::GPU_OurDecoupledLookback:
            impl_our_decoupled_lookback(d_input, d_output, num_elems);
            break;

        case InclusiveScanType::GPU_NvidiaDecoupledLookback:
            impl_nvidia_decoupled_lookback(d_input, d_output, num_elems);
            break;

        case InclusiveScanType::GPU_SimulateOptimalButIncorrect:
            if (cmd_args.check_) {
                print_warning("GPU_SimulateOptimalButIncorrect does not return "
                              "the correct answer; it merely simulates the "
                              "optimal timing with a memcpy!");
            }
            impl_simulate_optimal_but_incorrect_gpu(d_input,
                                                    d_output,
                                                    num_elems);
            break;
        case InclusiveScanType::GPU_CUBSimplified:
            impl_cub_simplified(d_input, d_output, num_elems);
            break;
        default:
            print_error("unrecognized scan type!");
            exit(-1);
        }

        const double end_time = get_time_in_seconds();
        print_duration(start_time, end_time);
    }

    // Copy output from device to host
    if (is_gpu_algorithm(cmd_args.type_)) {
        cuda_check(
            cudaHostAlloc(&h_output, array_size_in_bytes, cudaHostAllocDefault),
            "cudaHostAlloc(h_output)");
        assert(h_output != nullptr);

        cuda_check(cudaMemcpy(h_output,
                              d_output,
                              array_size_in_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(D2H)");
    }

    // Optionally check answer!
    if (cmd_args.check_) {
        int32_t ans = 0;
        for (size_t i = 0; i < num_elems; ++i) {
            ans += h_input[i];
            if (ans != h_output[i]) {
                std::cerr << "Error: output mismatch at " << i << ". Expected "
                          << ans << ", but got " << h_output[i] << std::endl;
                exit(-1);
            }
        }
    }

    // Free all resources
    if (is_gpu_algorithm(cmd_args.type_)) {
        cudaFree(d_input);
        cudaFree(d_input);
        cudaFreeHost(h_output);
    } else {
        free(h_output);
    }

    cudaDeviceReset();
    return 0;
}
