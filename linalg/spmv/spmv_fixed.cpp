//
// Created by niko on 11/14/19.
//

#include "spmv_fixed.h"

void fixed::spmv(const CSR& matrix, const std::vector<double>& x, std::vector<double>& y)  {
  assert(x.size() == static_cast<size_t>(matrix.num_cols()) && "Wrong dimension of x in A*x (CSR)");

#ifndef NDEBUG
  assert(y.size() == matrix.num_rows());
#else
  if (y.size() != matrix.num_rows()) {
    y.resize(matrix.num_rows());
  }
#endif

#pragma omp parallel for default(none) shared(matrix, x, y)
  for (size_t row = 0; row < matrix.rowptr().size() - 1; ++row) {
    double sum = 0.0;
    for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1); j++) {
      sum += matrix.values().at(j) * x.at(matrix.colidx().at(j));
    }
    y.at(row) = sum;
  }
}