#include <cassert>
#include <cstdint>
#include <iostream>

#include "uniform_random.hpp"

/// @note   This is a fake test file because I used the code under test as the
///         oracle. I also trust it because a "professional" wrote it...
int
main(void)
{
    uint32_t r = 0;
    foedus::assorted::UniformRandom urand0(0);
    foedus::assorted::UniformRandom urand13(13);

    r = urand0.uniform_within(0, 10);
    assert(r == 5);

    r = urand0.uniform_within(0, 10);
    assert(r == 9);

    r = urand13.uniform_within(0, 10);
    assert(r == 9);

    r = urand13.uniform_within(0, 10);
    assert(r == 10);

    std::cout << __FILE__ << ": OK!" << std::endl;

    return 0;
}

