//
// Created by niko on 11/14/19.
//

#ifndef CODE_SPMV_FIXED_H
#define CODE_SPMV_FIXED_H

#include <vector>

#include "../../matrix_formats/csr.hpp"

namespace fixed {
void spmv(const CSR &matrix, const std::vector<double> &x,
          std::vector<double> &y);
}
#endif // CODE_SPMV_FIXED_H
