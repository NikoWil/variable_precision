//
// Created by niko on 9/16/19.
//

#ifndef CODE_SPMV_H
#define CODE_SPMV_H

#include <vector>

#include "matrix_formats/csr.hpp"
#include "segmentation_char/segmentation_char.h"

template <int end, class OutputIt>
void spmv(const CSR& matrix, const std::vector<seg::Double_slice<0, end>>& x, OutputIt begin, OutputIt last) {
  assert(x.size() == matrix.num_cols() && "Wrong dimension of x in A*x (CSR)");
  assert(std::distance(begin, last) == matrix.num_rows());
  static_cast<void>(last);

  #pragma omp parallel for default(none) shared(x, begin, last, matrix)
  for (unsigned long row = 0; row < matrix.rowptr().size() - 1; row++) {
    double sum = 0.0;
    for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1); j++) {
      sum += matrix.values().at(j) * x.at(matrix.colidx().at(j)).to_double();
    }
    *(begin + row) = seg::Double_slice<0, end>{sum};
  }
}
#endif // CODE_SPMV_H
