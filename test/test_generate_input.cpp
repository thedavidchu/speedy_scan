#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "generate_input.hpp"

std::vector<int32_t> oracle = {
    10968269, 17747271, 13994639, 20656730, 11167724, 13990692, 5302276,
    16372957, 13512729, 7297279,  11832060, 15363417, 10865038, 15772614,
    2636686,  17754217, 4206308,  10867078, 15506736, 4864397,  1899483,
    21447043, 6307929,  17182986, 3562721,  16179744, 20584129, 18631394,
    19418375, 2433867,  11625943, 17617200, 7701287,  15087139, 17715627,
    13632862, 2408594,  13230249, 168527,   2144739,  14212417, 14106047,
    19852914, 19421662, 2028499,  13739030, 20272983, 18203814, 16250004,
    12189118, 11691641, 2692652,  799274,   6794621,  15883077, 11910627,
    14461193, 7878585,  4329378,  19056482, 1955195,  473457,   11494754,
    1935565,  7551539,  15501593, 4912245,  8560098,  13703956, 7071188,
    12803536, 12775530, 3829517,  7066631,  3837015,  19314113, 14157913,
    6040704,  8389128,  11165616, 7377041,  765290,   12298579, 8219547,
    7308075,  17214491, 10464839, 18962602, 401395,   7489493,  932034,
    14296130, 18249010, 8490886,  9702172,  19443829, 20955038, 5364846,
    19162909, 18594023};

void
print_vector(const std::vector<int32_t> &v)
{
    std::cout << "[";
    for (auto e : v) {
        std::cout << e << ", ";
    }
    std::cout << "]" << std::endl;
}

/// @note   This is a fake test file because I used the code under test as the
///         oracle. What is does verify, however, is determinism.
int
main(void)
{
    std::vector<int32_t> inputs = generate_input(100, 0);
    assert(inputs == oracle);
    std::cout << __FILE__ << ": OK!" << std::endl;
    return 0;
}