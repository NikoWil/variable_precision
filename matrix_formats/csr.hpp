//
// Created by niko on 6/6/19.
//

#ifndef CODE_CSR_HPP
#define CODE_CSR_HPP

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include "../util/util.hpp"

class CSR {
public:
explicit CSR(const std::vector<double>& values, const std::vector<int>& colidx, const std::vector<int>& rowptr, unsigned num_cols)
  :m_values(values), m_colidx(colidx), m_rowptr(rowptr), m_num_cols(num_cols)
  {
    assert(values.size() == colidx.size() && "CSR size of values and colidx must be same");
    for (const auto& e : colidx) {
      (void)e;
      assert(e >= 0 && static_cast<unsigned>(e) < num_cols && "CSR each colidx must fulfill 0 <= idx < num_cols");
    }
    for (const auto& e : rowptr) {

      (void)e;
      assert(e >= 0 && static_cast<size_t>(e) <= values.size() && "CSR rowptr entries must fulfill 0 <= e <= values.size()");
    }

    assert(std::is_sorted(rowptr.begin(), rowptr.end()) && "CSR rowptr must be sorted in ascending order");
    assert(static_cast<size_t>(rowptr.back()) == values.size() && "CSR rowptr last element must point at element after colidx/ values");
  }

// implements y = Ax
std::vector<double> spmv(const std::vector<double>& x) const;

static CSR empty() {
  return unit(0);
}

static CSR unit(unsigned n) {
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

static CSR symmetric(unsigned n, double density, std::mt19937 rng) {
  constexpr unsigned lower = 1;
  constexpr unsigned upper = 10000;
  static_assert(lower < upper, "");

  std::vector<std::vector<double>> matrix{n, std::vector<double>(n, 0)};
  std::uniform_real_distribution<> value_distrib(lower, upper);
  std::uniform_real_distribution<> diag_distrib(n * upper + 1, n * (lower + upper) + 1);
  std::uniform_int_distribution<> coord_distrib(0, n - 1);

  // make the matrix strictly diagonally dominated
  for (unsigned i = 0; i < n; ++i) {
    matrix.at(i).at(i) = diag_distrib(rng);
  }

  double elements = n * n;
  double non_zeroes = n;
  double curr_density = 0;
  while (curr_density < density) {
    unsigned row = coord_distrib(rng);
    unsigned col = coord_distrib(rng);

    if (row != col && matrix.at(row).at(col) == 0.) {
      const double val = value_distrib(rng) + 0.0625;
      matrix.at(row).at(col) = val;

      non_zeroes += 1;
      curr_density = non_zeroes / elements;
    }
  }

  std::vector<double> values{};
  std::vector<int> colidx{};
  std::vector<int> rowptr{0};
  for (size_t row = 0; row < matrix.size(); ++row) {
    for (size_t col = 0; col < matrix.at(row).size(); ++col) {
      if (matrix.at(row).at(col) != 0) {
        values.push_back(matrix.at(row).at(col));
        colidx.push_back(col);
      }
    }
    rowptr.push_back(values.size());
  }

  return CSR{values, colidx, rowptr, n};
}

static CSR diagonally_dominant(unsigned n, double density, std::mt19937 rng) {
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

const std::vector<double> & values() const {
  return m_values;
}

const std::vector<int> & colidx() const {
  return m_colidx;
}

const std::vector<int> & rowptr() const {
  return m_rowptr;
}

unsigned num_cols() const {
  return m_num_cols;
}

unsigned num_rows() const {
  return m_rowptr.size() - 1;
}

unsigned num_values() const {
  return m_values.size();
}

void print() const {
  print_vector(m_values, "m_values");
  print_vector(m_colidx, "m_colidx");
  print_vector(m_rowptr, "m_rowptr");
  std::cout << "m_num_cols: " << m_num_cols << std::endl;
  std::cout << "m_num_rows: " << this->num_rows() << std::endl;
}

private:
std::vector<double> m_values;
std::vector<int> m_colidx;
std::vector<int> m_rowptr;
unsigned m_num_cols;
};

#endif // CODE_CSR_HPP
