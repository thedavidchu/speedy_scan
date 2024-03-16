#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <sys/time.h>
#include <stdio.h>

#include "generate_input.hpp"

enum class InclusiveScanType { Baseline, DecoupledLookback, NvidiaScan };

static void
print_error(std::string msg)
{
    std::cerr << "Error: " << msg << std::endl;
}

struct CommandLineArguments {
    std::string exe_ = "?";
    // TODO(dchu)   Make these values into macros or somehow deduplicate the
    //              references to them!
    InclusiveScanType type_ = InclusiveScanType::Baseline;
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
            this->type_ = parse_inclusive_scan_type(argv[i]);
        } else if (matches_str(argv[i], {"--size", "-s"})) {
            ++i;
            this->size_ = parse_positive_int(argv[i]);
        } else if (matches_str(argv[i], {"--repeats", "-r"})) {
            ++i;
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
            print_error("unexpected argument");
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
                 "{baseline,decoupled-lookback,nvidia}. Default: baseline"
              << std::endl;
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
    if (matches_str(arg, {"baseline"})) {
        return InclusiveScanType::Baseline;
    } else if (matches_str(arg, {"decoupled-lookback"})) {
        return InclusiveScanType::DecoupledLookback;
    } else if (matches_str(arg, {"nvidia"})) {
        return InclusiveScanType::NvidiaScan;
    } else {
        print_error("unrecognized type");
        print_help();
        exit(-1);
    }
}

int
CommandLineArguments::parse_positive_int(char *arg)
{
    long x = 0;
    x = strtol(arg, NULL, 10);
    if (x > INT_MAX) {
        print_error("out of range");
        print_help();
        exit(-1);
    } else if (x == 0) {
        print_error("expecting non-zero value");
        print_help();
        exit(-1);
    }
    return static_cast<int>(x);
}

std::string
CommandLineArguments::inclusive_scan_type_to_input_string(
    InclusiveScanType type)
{
    switch (type) {
    case InclusiveScanType::Baseline:
        return "baseline";
    case InclusiveScanType::DecoupledLookback:
        return "decoupled-lookback";
    case InclusiveScanType::NvidiaScan:
        return "nvidia";
    default:
        print_error("unrecognized scan type!");
        exit(-1);
    }
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

    // Generate inputs
    int32_t *d_input = NULL;
    int32_t *d_output = NULL;
    std::vector<int32_t> h_input;
    if (cmd_args.debug_) {
        h_input = std::vector<int32_t>(cmd_args.size_, 1);
    } else {
        h_input = generate_input(cmd_args.size_, 0);
    }
    const size_t size = sizeof(*d_input) * cmd_args.size_;

    // Allocate memory
    cudaMalloc((void **)d_input, size);
    cudaMalloc((void **)d_output, size);

    // Copy input from host to device
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    switch (cmd_args.type_) {
    case InclusiveScanType::Baseline:
        cmd_args.print();
        for (int i = 0; i < cmd_args.repeats_; ++i) {
            double start_time = get_time_in_seconds();
            // TODO Call baseline kernel
            double end_time = get_time_in_seconds();
            print_duration(start_time, end_time);
        }
        break;
    case InclusiveScanType::DecoupledLookback:
        cmd_args.print();
        for (int i = 0; i < cmd_args.repeats_; ++i) {
            double start_time = get_time_in_seconds();
            // TODO Call decoupled lookback kernel
            double end_time = get_time_in_seconds();
            print_duration(start_time, end_time);
        }
        break;
    case InclusiveScanType::NvidiaScan:
        cmd_args.print();
        for (int i = 0; i < cmd_args.repeats_; ++i) {
            double start_time = get_time_in_seconds();
            // TODO Call nvidia kernel
            double end_time = get_time_in_seconds();
            print_duration(start_time, end_time);
        }
        break;
    default:
        print_error("unrecognized scan type!");
        exit(-1);
    }

    // Copy output from device to host
    int32_t *h_output = NULL;
    cudaHostAlloc(&h_output, size, cudaHostAllocDefault);
    assert(h_output != NULL);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyHostToDevice);

    // Optionally check answer!
    if (cmd_args.check_) {
        int32_t ans = 0;
        for (int i = 0; i < cmd_args.size_; ++i) {
            ans += h_input[i];
            assert(ans == h_output[i]);
        }
    }

    // Free all resources
    cudaFree(d_input);
    cudaFree(d_input);
    cudaFreeHost(h_output);

    cudaDeviceReset();
    return 0;
}
