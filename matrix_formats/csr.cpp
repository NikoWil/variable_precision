//
// Created by niko on 6/6/19.
//
#include <iostream>

#include "csr.hpp"

std::vector<double> CSR::spmv(const std::vector<double>& x) const {
  assert(x.size() == static_cast<size_t>(m_num_cols) && "Wrong dimension of x in A*x (CSR)");

  std::vector<double> y(num_rows());

  #pragma omp parallel for default(none) shared(x, y)
  for (unsigned long row = 0; row < m_rowptr.size() - 1; row++) {
    double sum = 0.0;
    for (auto j = m_rowptr.at(row); j < m_rowptr.at(row + 1); j++) {
      sum += m_values.at(j) * x.at(m_colidx.at(j));
    }
    y.at(row) = sum;
  }

  return y;
}

CSR CSR::empty() {
  return CSR::unit(0);
}

CSR CSR::unit(unsigned n) {
  std::vector<double> values(n, 1.);
  std::vector<int> colidx(n);
  std::vector<int> rowptr(n + 1);
  for (int i = 0; static_cast<unsigned>(i) < colidx.size(); i++) {
    colidx.at(i) = i;
    rowptr.at(i) = i;
  }
  rowptr.back() = static_cast<int>(n);

  return CSR{values, colidx, rowptr, n};
}

CSR CSR::diagonally_dominant(unsigned n, double density, std::mt19937 rng) {
  assert(density * n >= 1 && "Matrix needs at least 1 element per row");

  constexpr unsigned lower = 1;
  constexpr unsigned upper = 10000;
  static_assert(lower < upper, "");

  std::uniform_int_distribution<> index_distrib(0, n - 1);
  std::uniform_real_distribution<> value_distrib(lower, upper);
  // TODO: more variety in the diagonal?
  std::uniform_real_distribution<> diag_distrib(n * upper + 1, n * (lower + upper) + 1);

  std::vector<double> values;
  std::vector<int> colidx;
  std::vector<int> rowptr;
  rowptr.push_back(0);
  // construct all the rows
  for (unsigned row = 0; row < n; ++row) {
    std::vector<double> row_values{};
    // values for 1 row
    for (unsigned k = 0; k < density * n; ++k) {
      row_values.push_back(value_distrib(rng) + 0.0625);
    }

    std::set<int> colidx_set{};
    colidx_set.insert(row);
    while (colidx_set.size() < density * n) {
      colidx_set.insert(index_distrib(rng));
    }
    std::vector<int> row_colidx(colidx_set.size());
    std::copy(std::begin(colidx_set), std::end(colidx_set), std::begin(row_colidx));

    // insert diagonal dominant element
    auto diag_index = std::distance(std::begin(row_colidx), std::find(std::begin(row_colidx), std::end(row_colidx), row));
    row_values.at(diag_index) = diag_distrib(rng);

    // update matrix
    values.insert(std::end(values), std::begin(row_values), std::end(row_values));
    colidx.insert(std::end(colidx), std::begin(row_colidx), std::end(row_colidx));
    rowptr.push_back(values.size());
  }

  return CSR{values, colidx, rowptr, n};
}

CSR CSR::random(unsigned width, unsigned height, double density, std::mt19937 rng) {
  assert(width > 0 && "CSR::random Matrix width needs to be > 1");
  assert(height > 0 && "CSR::random Matrix height needs to be > 1");
  assert(density >= 0 && "CSR::random matrix density needs to be in [0, infinity)");

  std::vector<std::vector<double>> value_matrix{height, std::vector<double>(width, 0.)};

  std::uniform_int_distribution<> x_distrib(0, width - 1);
  std::uniform_int_distribution<> y_distrib(0, height - 1);
  std::uniform_real_distribution<> value_distrib(0.001, 100'000.);

  unsigned num_values{0};
  while (width * height * density < num_values) {
    auto x = x_distrib(rng);
    auto y = y_distrib(rng);
    if (value_matrix.at(y).at(x) == 0.) {
      continue;
    }
    auto val = value_distrib(rng);
    value_matrix.at(y).at(x) = val;
    ++num_values;
  }

  std::vector<double> values;
  std::vector<int> colidx;
  std::vector<int> rowptr;
  for (unsigned row = 0; row < height; ++row) {
    for (unsigned col = 0; col < width; ++col) {
      const auto pot_val = value_matrix.at(row).at(col);
      if (pot_val == 0.) {
        continue;
      }
      values.push_back(pot_val);
      colidx.push_back(col);
    }
    rowptr.push_back(values.size());
  }

  return CSR{values, colidx, rowptr, width};
}