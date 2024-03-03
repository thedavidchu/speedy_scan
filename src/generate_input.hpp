#include <cstdint>
#include <limits>
#include <vector>

#include "uniform_random.hpp"

/// @note   I inline this function not for performance reasons but rather so my
///         editor does not complain about a function definition in a header.
///         And this is a header out of shear laziness. The lesson is pretty
///         clear: laziness => speediness! XD
inline std::vector<int32_t>
generate_input(const int length, const uint64_t seed)
{
    std::vector<int32_t> input(length);
    if (length == 0) {
        return input;
    }

    // Initialize these after the guard clause, especially to avoid the division
    // by zero!
    foedus::assorted::UniformRandom urand(seed);
    const int32_t max_val = std::numeric_limits<int32_t>::max() / length;

    for (auto &e : input) {
        // NOTE There are two implicit casts from signed -> unsigned -> signed.
        e = urand.uniform_within(0, max_val);
    }

    return input;
}
