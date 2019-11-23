//
// Created by niko on 9/16/19.
//

#ifndef CODE_SPMV_H
#define CODE_SPMV_H

#include <vector>

#include "matrix_formats/csr.hpp"
#include "seg_char.h"

namespace seg_char {
template <int end>
void spmv(const CSR &matrix, const std::vector<seg::Double_slice<0, end>> &x,
          std::vector<seg::Double_slice<0, end>> &out) {
  assert(x.size() == matrix.num_cols() && "Wrong dimension of x in A*x (CSR)");
  assert(out.size() == matrix.num_rows());

  //#pragma omp parallel for default(none) shared(x, out, matrix)
  for (size_t row = 0; row < matrix.rowptr().size() - 1; row++) {
    double sum = 0.0;
    for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
         ++j) {
      sum += matrix.values().at(j) * x.at(matrix.colidx().at(j)).to_double();
    }
    out.at(row) = seg::Double_slice<0, end>{sum};
  }
}
}
#endif // CODE_SPMV_H
