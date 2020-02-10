//
// Created by niko on 11/14/19.
//

#ifndef CODE_POWERITERATION_H
#define CODE_POWERITERATION_H

#include <mpi.h>

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

namespace distributed {
    namespace fixed {
        std::pair<bool, int>
        power_iteration(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
                        MPI_Comm comm, int iteration_limit = 1000);
    }

    namespace seg_uint {
        std::pair<bool, int>
        power_iteration_2(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
                          MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);

        std::pair<bool, int>
        power_iteration_4(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
                          MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);

        std::pair<bool, int>
        power_iteration_6(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
                          MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);
    }
}
#endif // CODE_POWERITERATION_H
