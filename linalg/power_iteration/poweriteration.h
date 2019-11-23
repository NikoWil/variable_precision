//
// Created by niko on 11/14/19.
//

#ifndef CODE_POWERITERATION_H
#define CODE_POWERITERATION_H

#include "../../matrix_formats/csr.hpp"

namespace local {
/**
 * Perform power iteration on a single node using fixed precision.
 * This function serves as ground truth to check the correctness of other
 * power iteration implementations.
 */
std::pair<bool, int> power_iteration(const CSR &matrix,
                                         const std::vector<double> &x,
                                         std::vector<double> &curr,
                                         int iteration_limit = 1000);
}

#endif // CODE_POWERITERATION_H
