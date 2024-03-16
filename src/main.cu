#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

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
    unsigned int size_ = 1000;
    unsigned int repeats_ = 1;

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

    std::string
    inclusive_scan_type_to_input_string(InclusiveScanType type);

    unsigned
    parse_unsigned(char *arg);
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
            this->size_ = parse_unsigned(argv[i]);
        } else if (matches_str(argv[i], {"--repeats", "-r"})) {
            ++i;
            this->repeats_ = parse_unsigned(argv[i]);
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
              << ")" << std::endl;
}

void
CommandLineArguments::print_help()
{
    std::cout << "Usage: " << this->exe_
              << " [-t <scan-type>] [-s <input-size>] [-r <repetitions>]"
              << std::endl;
    std::cout << "    -t, --type: scan type, "
                 "{baseline,decoupled-lookback,nvidia}. Default: baseline"
              << std::endl;
    std::cout << "    -s, --size: number of input elements, 1..= "
                 "~1_000_000_000. Default: 1000"
              << std::endl;
    std::cout << "    -r, --repeats: number of times the test is repeated, "
                 "0..=MAX_UNSIGNED_INT. Default: 1"
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

unsigned
CommandLineArguments::parse_unsigned(char *arg)
{
    unsigned long x = 0;
    errno = 0;
    x = strtoul(arg, NULL, 10);
    if (x > UINT_MAX || x == ULONG_MAX && errno == ERANGE) {
        print_error("out of range");
        print_help();
        exit(-1);
    } else if (x == 0) {
        print_error("expecting non-zero value");
        print_help();
        exit(-1);
    }
    return x;
}

int
main(int argc, char *argv[])
{
    CommandLineArguments cmd_args(argc, argv);

    switch (cmd_args.type_) {
    case InclusiveScanType::Baseline:
        cmd_args.print();
        break;
    case InclusiveScanType::DecoupledLookback:
        cmd_args.print();
        break;
    case InclusiveScanType::NvidiaScan:
        cmd_args.print();
        break;
    default:
        print_error("unrecognized scan type!");
        exit(-1);
    }
    return 0;
}
