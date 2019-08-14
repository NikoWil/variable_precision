//
// Created by niko on 6/6/19.
//
#include <iostream>
#include <omp.h>

#include "csr.hpp"

std::vector<double> CSR::spmv(const std::vector<double>& x) const {
  assert(x.size() == static_cast<size_t>(m_num_cols) && "Wrong dimension of x in A*x (CSR)");

  std::vector<double> y(num_rows());

  omp_set_num_threads(8);

  #pragma omp parallel for default(none) shared(x, y, std::cout)
  for (unsigned long row = 0; row < m_rowptr.size() - 1; row++) {
    if (row == 0) {
      std::cout << omp_get_num_threads() << std::endl;
    }
    double sum = 0.0;
    for (auto j = m_rowptr.at(row); j < m_rowptr.at(row + 1); j++) {
      sum += m_values.at(j) * x.at(m_colidx.at(j));
    }
    y.at(row) = sum;
  }

  return y;
}