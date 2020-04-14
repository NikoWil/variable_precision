//
// Created by khondar on 17.03.20.
//

#include "pr_spmv.h"
#include "../segmentation/seg_uint.h"


void pagerank::fixed::spmv(const CSR &matrix, const std::vector<double> &x, std::vector<double> &y, double c) {
    assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
           "pagerank::fixed::spmv Wrong dimension of x or A in A*x");
    assert(y.size() == matrix.num_rows() &&
           "pagerank::fixed::spmv Wrong dimension of y in Ax = y");

    int num_rows = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, num_rows, c)
    for (int row = 0; row < num_rows; ++row) {
        double sum = 0.0;   // calculate M * rank
        for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1); j++) {
            sum += matrix.values().at(j) * x.at(matrix.colidx().at(j));
        }

        y.at(row) = c * sum + (1. - c) / static_cast<double>(matrix.num_cols());
    }
}

void
pagerank::seg::spmv_2(const CSR &matrix, const std::vector<std::uint16_t> &x, std::vector<std::uint16_t> &y, double c) {
    assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
           "pagerank::seg::spmv_2 Wrong dimension of x or A in A*x");
    assert(y.size() == matrix.num_rows() &&
           "pagerank::seg::spmv_2 Wrong dimension of y in Ax = y");

    int num_rows = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, num_rows, c)
    for (int row = 0; row < num_rows; ++row) {
        double sum = 0.0;
        for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1); j++) {
            double x_at;
            const uint16_t *x_bytes = &x.at(matrix.colidx().at(j));
            seg_uint::read_2(x_bytes, &x_at);
            sum += matrix.values().at(j) * x_at;
        }
        sum = c * sum + (1. - c) / static_cast<double>(num_rows);
        std::uint16_t rep;
        seg_uint::write_2(&rep, &sum);
        y.at(row) = rep;
//        seg_uint::write_2(&y.at(row), &sum);
    }
}

void
pagerank::seg::spmv_4(const CSR &matrix, const std::vector<std::uint32_t> &x, std::vector<std::uint32_t> &y, double c) {
    assert(x.size() == static_cast<size_t>(matrix.num_cols()) &&
           "pagerank::seg::spmv_4 Wrong dimension of x or A in A*x");
    assert(y.size() == matrix.num_rows() &&
           "pagerank::seg::spmv_4 Wrong dimension of y in Ax = y");

    int num_rows = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, num_rows, c)
    for (int row = 0; row < num_rows; ++row) {
        double sum = 0.0;
        for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1); j++) {
            double x_at;
            const uint32_t *x_bytes = &x.at(matrix.colidx().at(j));
            seg_uint::read_4(x_bytes, &x_at);
            sum += matrix.values().at(j) * x_at;
        }
        sum = c * sum + (1 - c) / static_cast<double>(matrix.num_cols());
        seg_uint::write_4(&y.at(row), &sum);
    }//*/
}

void
pagerank::seg::spmv_6(const CSR &matrix, const std::vector<std::uint16_t> &x, std::vector<std::uint16_t> &y, double c) {
    assert(x.size() == 3 * static_cast<size_t>(matrix.num_cols()) &&
           "pagerank::seg::spmv_6 Wrong dimension of x or A in A*x");
    assert(y.size() == 3 * matrix.num_rows() &&
           "pagerank::seg::spmv_6 Wrong dimension of y in Ax = y");

    int num_rows = static_cast<int>(matrix.rowptr().size() - 1);
#pragma omp parallel for default(none) shared(matrix, x, y, num_rows, c)
    for (int row = 0; row < num_rows; ++row) {
        double sum = 0.0;
        for (auto j = matrix.rowptr().at(row); j < matrix.rowptr().at(row + 1); j++) {
            double x_at;
            const uint16_t *x_bytes = &x.at(3 * matrix.colidx().at(j));
            seg_uint::read_6(x_bytes, &x_at);
            sum += matrix.values().at(j) * x_at;
        }
        sum = c * sum + (1 - c) / static_cast<double>(matrix.num_cols());
        seg_uint::write_6(&y.at(3 * row), &sum);
    }
}