//
// Created by khondar on 17.03.20.
//

#ifndef CODE_PR_SPMV_H
#define CODE_PR_SPMV_H

#include <vector>
#include "../matrix_formats/csr.hpp"

namespace pagerank {
    namespace fixed {
        void spmv(const CSR &matrix, const std::vector<double> &x, std::vector<double> &y, double c);
    }

    namespace seg {
        void spmv_2(const CSR &matrix, const std::vector<std::uint16_t> &x, std::vector<std::uint16_t> &y, double c);

        void spmv_4(const CSR &matrix, const std::vector<std::uint32_t> &x, std::vector<std::uint32_t> &y, double c);

        void spmv_6(const CSR &matrix, const std::vector<std::uint16_t> &x, std::vector<std::uint16_t> &y, double c);
    }
}

#endif //CODE_PR_SPMV_H
