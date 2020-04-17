//
// Created by khondar on 17.03.20.
//

#ifndef CODE_PAGERANK_H
#define CODE_PAGERANK_H

#include <array>
#include <cstdint>
#include <mpi.h>
#include <utility>
#include <vector>

#include "../matrix_formats/csr.hpp"

namespace pagerank {
    struct pr_meta {
        bool converged;
        int used_iterations;
        std::int64_t prep_timing;
        std::vector<std::int64_t> spmv_timings;
        std::vector<std::int64_t> agv_timings;
        std::vector<std::int64_t> overhead_timings;
    };

    namespace local {
        std::pair<bool, int>
        pagerank(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                 int iteration_limit = 1000);
    }
    namespace fixed {
        pr_meta pagerank(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                         MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);
    }

    namespace seg {
        pr_meta
        pagerank_2(const CSR &matrix, const std::vector<std::uint16_t> &initial, std::vector<std::uint16_t> &result,
                   double c, MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);

        pr_meta
        pagerank_4(const CSR &matrix, const std::vector<std::uint32_t> &initial, std::vector<std::uint32_t> &result,
                   double c, MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);

        pr_meta
        pagerank_6(const CSR &matrix, const std::vector<std::uint16_t> &initial, std::vector<std::uint16_t> &result,
                   double c, MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);
    }

    namespace variable {
        std::array<pr_meta, 4>
        pagerank_2_4_6_8(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                         MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);

        std::array<pr_meta, 3>
        pagerank_4_6_8(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                       MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);

        std::array<pr_meta, 2>
        pagerank_6_8(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                     MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit = 1000);
    }
}

#endif //CODE_PAGERANK_H
