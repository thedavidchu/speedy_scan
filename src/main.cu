#include <iostream>
#include <string>
#include <cstring>

enum class InclusiveScanType {
    Baseline, DecoupledLookback, NvidiaScan
};

struct CommandLineArguments {
    InclusiveScanType type_;
    unsigned int size_;
    unsigned int repeats_;

    CommandLineArguments(int argc, char *argv[]) {
        // Set defaults
        this->type_ = InclusiveScanType::Baseline;
        this->size_ = 1000;
        this->repeats_ = 1;

        // Set parameters based on user arguments
        for (int i = 1; i < argc; ++i) {
            if (matches_flag(argv[i], {"-t", "--type"})) {
                ++i;
                this->type_ = parse_type(argv[i]);
            } else if (matches_flag(argv[i], {"--size", "-s"})) {
                ++i;
                this->size_ = parse_unsigned(argv[i]);
            } else if (matches_flag(argv[i], {"--repeats", "-r"})) {
                ++i;
                this->repeats_ = parse_unsigned(argv[i]);
            } else if (matches_flag(argv[i], {"--help", "-h"})) {
                print_help();
                exit(0);
            } else
                // TODO(dchu): print out the unexpected argument that we received
                print_error("unexpected argument");
                print_help();
                exit(-1);
            }
        }
    }

private:
    void
    matches_flag(char *arg, std::vector<std::string> flags)
    {
        for (auto flag : flags) {
            if (strcmp(arg, flag.c_str()) == 0) {
                return true;
            }
        }
        return false;
    }

    InclusiveScanType
    parse_type(char *arg)
    {
        if (matches_flag(arg, {"baseline"})) {
            return InclusiveScanType::Baseline;
        } else if (matches_flag(arg, {})) {
            return InclusiveScanType::DecoupledLookback;
        } else if (matches_flag(arg, {})) {
            return InclusiveScanType::NvidiaScan;
        } else {
            print_error("unrecognized type");
            print_help();
            exit(-1);
        }
    }

    unsigned
    parse_unsigned(char *arg)
    {
        return strtoul(arg, NULL, 10);
    }

    void
    print_error(std::string msg)
    {
        std::cerr << "Error: " << msg << std::endl;
    }

    void
    print_help(std::string exec_name)
    {
        std::cout << "Usage: " << exec_name << " [-t <scan-type>] [] []" << std::endl;
        std::cout << "    -t, --type: scan type, {baseline,decoupled-lookback,nvidia}. Default: baseline" << std::endl;
        std::cout << "    -s, --size: number of input elements, 1..=1_000_000_000. Default: 1000" << std::endl;
        std::cout << "    -r, --repeats: number of times the test is repeated, 0..=MAX_UNSIGNED_INT. Default: 1" << std::endl;
        std::cout << "    -h, --help: print this help message. Overrides all else!" << std::endl;
    }
};

int
main(int argc, char *argv[])
{
    CommandLineArguments cmd_args(argc, argv);
    switch (cmd_args.scan_type_) {
    case InclusiveScanType::Baseline:
        break;
    case InclusiveScanType::DecoupledLookback:
        break;
    case InclusiveScanType::NvidiaScan:
        break;
    default:
        exit(-1);
    }
    return 0;
}
