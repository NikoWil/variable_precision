//
// Created by niko on 11/14/19.
//

#include "spmv_fixed.h"
#include "../segmentation/seg_uint.h"

void fixed::spmv(const CSR &matrix, const std::vector<double> &x, std::vector<double> &y) {
    assert(x.size() == static_cast<size_t>(matrix.num_cols()) && "fixed::spmv wrong dimension of x in Ax = y");
    assert(y.size() == matrix.num_rows() && "fixed::spmv y has wrong size in Ax = y");

    int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
    for (int row = 0; row < upper; ++row) {
        double sum = 0.0;
        for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1); j++) {
            sum += matrix.values().at(j) * x.at(matrix.colidx().at(j));
        }
        y.at(row) = sum;
    }
}

namespace seg_uint {
/*
namespace pre_convert {
void spmv_2(const CSR& matrix, const std::vector<uint16_t> &x,
            std::vector<uint16_t> &y) {
  assert(x.size() == static_cast<size_t>(matrix.num_cols()) && "seg_uint::pre_convert::spmv_2 Wrong dimension of x in Ax = y");
  assert(y.size() == matrix.num_rows() && "seg_uint::pre_convert::spmv_2 y has wrong size in Ax = y");

  std::vector<double> x_double(x.size());
  for (size_t i{0}; i < x.size(); ++i) {
    read_2(&x[i], &x_double[i]);
  }
  std::vector<double> y_double(y.size());

  fixed::spmv(matrix, x_double, y_double);
  assert(y.size() == y_double.size() && "seg_uint::pre_convert::spmv_2 size of y_double was changed by fixed::spmv");

  for (size_t i{0}; i < y.size(); ++i) {
    write_2(&y[i], &y_double[i]);
  }
}

void spmv_4(const CSR& matrix, const std::vector<uint32_t> &x,
            std::vector<uint32_t> &y) {
  assert(x.size() == static_cast<size_t>(matrix.num_cols()) && "seg_uint::pre_convert::spmv_4 Wrong dimension of x in Ax = y");
  assert(y.size() == matrix.num_rows() && "seg_uint::pre_convert::spmv_4 y has wrong size in Ax = y");

  std::vector<double> x_double(x.size());
  for (size_t i{0}; i < x.size(); ++i) {
    read_4(&x[i], &x_double[i]);
  }
  std::vector<double> y_double(y.size());

  fixed::spmv(matrix, x_double, y_double);
  assert(y.size() == y_double.size() && "seg_uint::pre_convert::spmv_4 size of y_double was changed by fixed::spmv");

  for (size_t i{0}; i < y.size(); ++i) {
    write_4(&y[i], &y_double[i]);
  }
}

void spmv_6(const CSR& matrix, const std::vector<uint16_t> &x,
            std::vector<uint16_t> &y) {
  assert(x.size() == 3 * static_cast<size_t>(matrix.num_cols()) && "seg_uint::pre_convert::spmv_6 Wrong dimension of x in Ax = y");
  assert(y.size() == 3 * matrix.num_rows() && "seg_uint::pre_convert::spmv_6 y has wrong size in Ax = y");
  assert(x.size() % 3 == 0 && "seg_uint::pre_convert::spmv_6 x in Ax = y has invalid number of elements");
  assert(y.size() % 3 == 0 && "seg_uint::pre_convert::spmv_6 y in Ax = y has invalid number of elements");

  std::vector<double> x_double(matrix.num_cols());
  for (size_t i{0}; i < matrix.num_cols(); ++i) {
    read_6(&x.at(i * 3), &x_double.at(i));
  }
  std::vector<double> y_double(matrix.num_rows());

  fixed::spmv(matrix, x_double, y_double);
  assert(matrix.num_rows() == y_double.size() && "seg_uint::pre_convert::spmv_3 size of y_double was changed by fixed::spmv");

  for (size_t i{0}; i < matrix.num_rows(); ++i) {
    write_6(&y.at(i * 3), &y_double.at(i));
  }
}
}
*/

    namespace calc_convert {
        void spmv_2(const CSR &matrix, const std::vector<uint16_t> &x,
                    std::vector<uint16_t> &y) {
            assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
                   "seg_uint::spmv_2 Wrong dimension of x or A in A*x");

            assert(y.size() == matrix.num_rows() &&
                   "seg_uint::spmv_2 Wrong dimension of y in Ax = y");

            int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
            for (int row = 0; row < upper; ++row) {
                double sum = 0.0;
                for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
                     j++) {
                    double x_at;
                    const uint16_t *x_bytes = &x.at(matrix.colidx().at(j));
                    seg_uint::read_2(x_bytes, &x_at);

                    sum += matrix.values().at(j) * x_at;
                }
                seg_uint::write_2(&y.at(row), &sum);
            }
        }

        void spmv_4(const CSR &matrix, const std::vector<uint32_t> &x,
                    std::vector<uint32_t> &y) {
            assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
                   "seg_uint::spmv_4 Wrong dimension of x or A in A*x");

            assert(y.size() == matrix.num_rows() &&
                   "seg_uint::spmv_4 Wrong dimension of y in Ax = y");

            int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
            for (int row = 0; row < upper; ++row) {
                double sum = 0.0;
                for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
                     j++) {
                    double x_at;
                    const uint32_t *x_bytes = &x.at(matrix.colidx().at(j));
                    seg_uint::read_4(x_bytes, &x_at);

                    sum += matrix.values().at(j) * x_at;
                }
                seg_uint::write_4(&y.at(row), &sum);
            }
        }

        void spmv_6(const CSR &matrix, const std::vector<uint16_t> &x,
                    std::vector<uint16_t> &y) {
            assert(x.size() == 3 * static_cast<size_t>(matrix.num_cols()) &&
                   "seg_uint::spmv_2 Wrong dimension of x or A in A*x");

            assert(y.size() == 3 * matrix.num_rows() &&
                   "seg_uint::spmv_2 Wrong dimension of y in Ax = y");

            int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
            for (int row = 0; row < upper; ++row) {
                double sum = 0.0;
                for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
                     j++) {
                    double x_at;
                    const uint16_t *x_bytes = &x.at(3 * matrix.colidx().at(j));
                    seg_uint::read_6(x_bytes, &x_at);

                    sum += matrix.values().at(j) * x_at;
                }
                seg_uint::write_6(&y.at(3 * row), &sum);
            }
        }
    }

    namespace out_convert {
        void spmv_2(const CSR &matrix, const std::vector<double> &x,
                    std::vector<uint16_t> &y) {
            assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
                   "seg_uint::out_convert::spmv_2 Wrong dimension of x or A in A*x");

            assert(y.size() == matrix.num_rows() &&
                   "seg_uint::out_convert::spmv_2 Wrong dimension of y in Ax = y");

            int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
            for (int row = 0; row < upper; ++row) {
                double sum = 0.0;
                for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
                     j++) {
                    sum += matrix.values().at(j) * x.at(matrix.colidx().at(j));
                }
                seg_uint::write_2(&y.at(row), &sum);
            }
        }

        void spmv_4(const CSR &matrix, const std::vector<double> &x,
                    std::vector<uint32_t> &y) {
            assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
                   "seg_uint::out_convert::spmv_4 Wrong dimension of x or A in A*x");

            assert(y.size() == matrix.num_rows() &&
                   "seg_uint::out_convert::spmv_4 Wrong dimension of y in Ax = y");

            int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
            for (int row = 0; row < upper; ++row) {
                double sum = 0.0;
                for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
                     j++) {
                    sum += matrix.values().at(j) * x.at(matrix.colidx().at(j));
                }
                seg_uint::write_4(&y.at(row), &sum);
            }
        }

        void spmv_6(const CSR &matrix, const std::vector<double> &x,
                    std::vector<uint16_t> &y) {
            assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
                   "seg_uint::out_convert::spmv_6 Wrong dimension of x or A in A*x");

            assert(y.size() == 3 * matrix.num_rows() &&
                   "seg_uint::out_convert::spmv_6 Wrong dimension of y in Ax = y");

            int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
            for (int row = 0; row < upper; ++row) {
                double sum = 0.0;
                for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
                     j++) {
                    sum += matrix.values().at(j) * x.at(matrix.colidx().at(j));
                }
                seg_uint::write_6(&y.at(3 * row), &sum);
            }
        }
    }
}
