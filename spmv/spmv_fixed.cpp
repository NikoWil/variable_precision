//
// Created by niko on 11/14/19.
//

#include "spmv_fixed.h"
#include "../segmentation/seg_uint.h"

void fixed::spmv(const CSR &matrix, const std::vector<double> &x, std::vector<double> &__restrict__ y) {
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
    namespace calc_convert {
        void spmv_2(const CSR &matrix, const std::vector<uint16_t> &x,
                    std::vector<uint16_t> &__restrict__ y) {
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
                    const uint16_t *x_bytes = &x.at(matrix.colidx().at(j));
                    const double x_at = seg_uint::read_2(x_bytes);

                    sum += matrix.values().at(j) * x_at;
                }
                y.at(row) = seg_uint::write_2(&sum);
            }
        }

        void spmv_4(const CSR &matrix, const std::vector<uint32_t> &x,
                    std::vector<uint32_t> &__restrict__ y) {
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
                    const uint32_t *x_bytes = &x.at(matrix.colidx().at(j));
                    const double x_at = seg_uint::read_4(x_bytes);

                    sum += matrix.values().at(j) * x_at;
                }
                y.at(row) = seg_uint::write_4(&sum);
            }
        }

        void spmv_6(const CSR &matrix, const std::vector<std::array<std::uint16_t, 3>> &x,
                    std::vector<std::array<std::uint16_t, 3>> &__restrict__ y) {
            assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
                   "seg_uint::spmv_6 Wrong dimension of x or A in A*x");

            assert(y.size() == matrix.num_rows() &&
                   "seg_uint::spmv_6 Wrong dimension of y in Ax = y");

            int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
            for (int row = 0; row < upper; ++row) {
                double sum = 0.0;
                for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
                     j++) {
                    const double x_at = seg_uint::read_6(x.at(matrix.colidx().at(j)));

                    sum += matrix.values().at(j) * x_at;
                }
                y.at(row) = seg_uint::write_6(&sum);
            }
        }
    }

    namespace out_convert {
        void spmv_2(const CSR &matrix, const std::vector<double> &x,
                    std::vector<uint16_t> &__restrict__ y) {
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
                y.at(row) = seg_uint::write_2(&sum);
            }
        }

        void spmv_4(const CSR &matrix, const std::vector<double> &x,
                    std::vector<uint32_t> &__restrict__ y) {
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
                y.at(row) = seg_uint::write_4(&sum);
            }
        }

        void spmv_6(const CSR &matrix, const std::vector<double> &x,
                    std::vector<std::array<std::uint16_t, 3>> &__restrict__ y) {
            assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
                   "seg_uint::out_convert::spmv_6 Wrong dimension of x or A in A*x");

            assert(y.size() == matrix.num_rows() &&
                   "seg_uint::out_convert::spmv_6 Wrong dimension of y in Ax = y");

            int upper = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, upper)
            for (int row = 0; row < upper; ++row) {
                double sum = 0.0;
                for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1);
                     j++) {
                    sum += matrix.values().at(j) * x.at(matrix.colidx().at(j));
                }
                y.at(row) = seg_uint::write_6(&sum);
            }
        }
    }
}
