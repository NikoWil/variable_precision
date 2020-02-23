//
// Created by niko on 11/14/19.
//

#ifndef CODE_SPMV_FIXED_H
#define CODE_SPMV_FIXED_H

#include <vector>

#include "../matrix_formats/csr.hpp"

namespace fixed {
    void spmv(const CSR &matrix, const std::vector<double> &x,
              std::vector<double> &y);
}

namespace seg_uint {
/*
namespace pre_convert {
void spmv_2(const CSR& matrix, const std::vector<uint16_t > &x,
            std::vector<uint16_t > &y);

void spmv_4(const CSR& matrix, const std::vector<uint32_t > &x,
            std::vector<uint32_t > &y);

void spmv_6(const CSR& matrix, const std::vector<uint16_t > &x,
            std::vector<uint16_t> &y);
}
*/

    namespace calc_convert {
        void spmv_2(const CSR &matrix, const std::vector<uint16_t> &x,
                    std::vector<uint16_t> &y);

        void spmv_4(const CSR &matrix, const std::vector<uint32_t> &x,
                    std::vector<uint32_t> &y);

        void spmv_6(const CSR &matrix, const std::vector<uint16_t> &x,
                    std::vector<uint16_t> &y);
    }

    namespace out_convert {
        void spmv_2(const CSR &matrix, const std::vector<double> &x,
                    std::vector<uint16_t> &y);

        void spmv_4(const CSR &matrix, const std::vector<double> &x,
                    std::vector<uint32_t> &y);

        void spmv_6(const CSR &matrix, const std::vector<double> &x,
                    std::vector<uint16_t> &y);
    }
}

#endif // CODE_SPMV_FIXED_H
