//
// Created by khondar on 17.03.20.
//

#ifndef CODE_PAGERANK_H
#define CODE_PAGERANK_H

#include <mpi.h>
#include <utility>
#include <vector>

#include "../matrix_formats/csr.hpp"

namespace pagerank {
    namespace fixed {
        std::pair<bool, int>
        pagerank(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                 MPI_Comm comm, int iteration_limit = 1000);
    }

    namespace seg {
        std::pair<bool, int>
        pagerank_2(const CSR &matrix, const std::vector<std::uint16_t> &initial, std::vector<std::uint16_t> &result,
                   double c, MPI_Comm comm, int iteration_limit = 1000);

        std::pair<bool, int>
        pagerank_4(const CSR &matrix, const std::vector<std::uint32_t> &initial, std::vector<std::uint32_t> &result,
                   double c, MPI_Comm comm, int iteration_limit = 1000);

        std::pair<bool, int>
        pagerank_6(const CSR &matrix, const std::vector<std::uint16_t> &initial, std::vector<std::uint16_t> &result,
                   double c, MPI_Comm comm, int iteration_limit = 1000);
    }
}

#endif //CODE_PAGERANK_H
