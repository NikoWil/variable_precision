//
// Created by niko on 6/6/19.
//
#include <iostream>

#include "csr.hpp"
#include "../communication.h"

void CSR::concat_vertical(const CSR &other) {
    assert(this->num_cols() == other.num_cols() &&
           "CSR::concat_vertical both matrices need to have same number of columns");
    /**
     * 1. generate copy of other.rowptr
     * 2. add this.rowptr.end to each element of other.rowptr
     * 3. concatenate this.values and other.values
     *                this.colidx and other.colidx
     *                this.rowptr and other.rowptr[1..]
     */
    std::vector<int> rowptr_extension;
    rowptr_extension.reserve(other.m_rowptr.size() - 1);
    // Skip the first element as it's a duplicate between both elements
    rowptr_extension.insert(rowptr_extension.begin(), other.m_rowptr.begin() + 1, other.m_rowptr.end());
    for (auto &e : rowptr_extension) {
        e += this->m_rowptr.back();
    }
    this->m_rowptr.reserve(this->m_rowptr.size() + rowptr_extension.size());
    this->m_rowptr.insert(this->m_rowptr.end(), rowptr_extension.begin(), rowptr_extension.end());

    this->m_colidx.reserve(this->m_colidx.size() + other.m_colidx.size());
    this->m_colidx.insert(m_colidx.end(), other.m_colidx.begin(), other.m_colidx.end());
    this->m_values.reserve(this->m_values.size() + other.m_values.size());
    this->m_values.insert(this->m_values.begin(), other.m_values.begin(), other.m_values.end());
}

void CSR::concat_horizontal(const CSR &other) {
    assert(this->num_rows() == other.num_rows() &&
           "CSR::concat_horizontal both matrices need to have same number of rows");

    std::vector<int> new_rowptr;
    new_rowptr.reserve(this->num_rows());
    for (std::size_t i{0}; i < this->m_rowptr.size(); ++i) {
        new_rowptr.push_back(this->m_rowptr.at(i) + other.m_rowptr.at(i));
    }

    std::vector<int> colidx_extension;
    colidx_extension.reserve(other.m_colidx.size());
    for (const auto e : other.m_colidx) {
        colidx_extension.push_back(e + this->m_num_cols);
    }

    std::vector<double> new_values;
    new_values.reserve(this->m_values.size() + other.m_values.size());
    std::vector<int> new_colidx;
    new_colidx.reserve(this->m_colidx.size() + other.m_colidx.size());
    for (std::size_t row{0}; row < this->num_rows(); ++row) {
        new_values.insert(new_values.end(),
                          this->m_values.begin() + this->m_rowptr.at(row),
                          this->m_values.begin() + this->m_rowptr.at(row + 1));
        new_values.insert(new_values.end(),
                          other.m_values.begin() + other.rowptr().at(row),
                          other.m_values.begin() + other.m_rowptr.at(row + 1));
        new_colidx.insert(new_colidx.end(),
                          this->m_colidx.begin() + this->m_rowptr.at(row),
                          this->m_colidx.begin() + this->m_rowptr.at(row + 1));
        new_colidx.insert(new_colidx.end(),
                          colidx_extension.begin() + other.m_rowptr.at(row),
                          colidx_extension.begin() + other.m_rowptr.at(row + 1));
    }

    this->m_values = new_values;
    this->m_colidx = new_colidx;
    this->m_rowptr = new_rowptr;
    this->m_num_cols += other.m_num_cols;
}

/**
 * 1. count #values per column, use as new rowptr
 * 2. iterate through all values, find their old colidx and use the new rowptr to offset them into the right new place
 * @param matrix
 * @return
 */
CSR CSR::transpose(const CSR &matrix) {
    assert(matrix.num_rows() == matrix.num_rows() && "CSR::transpose expectes the input matrix to be square");
    const int n = static_cast<int>(matrix.num_cols());
    std::vector<double> values(matrix.m_values.size());
    std::vector<int> colidx(matrix.m_colidx.size());
    std::vector<int> rowptr(n + 1);

    std::vector<int> colcnts(n, 0);
    for (const auto c : matrix.colidx()) {
        colcnts.at(c) += 1;
    }

    rowptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        rowptr[i + 1] = rowptr[i] + colcnts[i];
    }
    assert(static_cast<size_t>(rowptr[n]) == values.size() && "CSR::transpose assert rowptr[n] == values.size()");

    std::vector<int> offsets(n);
    std::copy(rowptr.begin(), rowptr.end() - 1, offsets.begin());

    int row = 0;
    for (unsigned i = 0; i < values.size(); ++i) {
        if (i == static_cast<unsigned>(matrix.m_rowptr[row + 1])) {
            row += 1;
        }

        const int idx = offsets[matrix.m_colidx[i]];
        values[idx] = matrix.m_values[i];
        colidx[idx] = row;

        offsets[matrix.m_colidx[i]] += 1;
    }

    return CSR{values, colidx, rowptr, matrix.num_rows()};
}

CSR CSR::empty(std::size_t width, std::size_t height) {
    std::vector<double> values;
    std::vector<int> colidx;
    std::vector<int> rowptr(height + 1, 0);
    return CSR{values, colidx, rowptr, width};
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

CSR CSR::row_stochastic(unsigned width, unsigned height, double density, std::mt19937 &rng) {
    assert(0 <= density && density <= 1 && "CSR::row_stochastic density needs to be in interval [0, 1]");
    auto vals_per_row = static_cast<unsigned>(density * width);
    assert((vals_per_row >= 1) && "CSR::row_stochastic needs density that guarantees at least 1 value per row");
    /*assert(index_distribution.a() == 0 && "CSR::row_stochastic index_distribution.a() must be 0");
    assert(static_cast<unsigned>(index_distribution.b()) == width - 1 &&
           "CSR::row_stochastic index_distribution.a() must be width - 1");*/

    std::vector<double> values;
    values.reserve(height * vals_per_row);
    std::vector<int> colidx;
    colidx.reserve(height * vals_per_row);
    std::vector<int> rowptr;
    rowptr.reserve(height + 1);
    rowptr.push_back(0);

    double normed_value =
            1. / static_cast<double>(vals_per_row); // each entry is of size 1 / vals_per_row, s.t. sum(row) = 1.

    std::uniform_int_distribution<> index_distribution(0, width - 1);
    // generate & append one more row
    for (unsigned i = 0; i < height; ++i) {
        std::vector<double> new_values(vals_per_row, normed_value);
        std::set<int> indices;
        while (indices.size() < vals_per_row) {
            indices.insert(index_distribution(rng));
        }
        std::vector<int> new_colidx(indices.begin(), indices.end());
        std::sort(new_colidx.begin(), new_colidx.end());

        values.insert(values.end(), new_values.begin(), new_values.end());
        colidx.insert(colidx.end(), new_colidx.begin(), new_colidx.end());
        rowptr.push_back(colidx.size());
    }

    return CSR{values, colidx, rowptr, width};
}

CSR CSR::row_stochastic(unsigned int n, double density, std::mt19937 rng) {
    assert(0 <= density && density <= 1 && "CSR::row_stochastic density needs to be in interval [0, 1]");
    assert((density * n >= 1) && "CSR::row_stochastic needs density that guarantees at least 1 value per row");

    return row_stochastic(n, n, density, rng);
}

CSR
CSR::distributed_column_stochastic(std::size_t n, double density, std::mt19937 rng, std::size_t num_steps, MPI_Comm comm,
                                   int root) {
    const auto width = n;
    const auto height = n;
    const auto block_size = height / num_steps;
    auto remaining_height = height;

    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    CSR initial = CSR::empty(0, height);
    CSR matrix_builder = distribute_matrix(initial, comm, root);

    while (remaining_height > 0) {
        const auto next_height = std::min(block_size, remaining_height);
        CSR matrix_block;
        if (comm_rank == root) {
            matrix_block = row_stochastic(width, next_height, density, rng);
            matrix_block = transpose(matrix_block);
        }
        CSR matrix_block_slice = distribute_matrix(matrix_block, comm, root);
        matrix_builder.concat_horizontal(matrix_block_slice);

        remaining_height -= next_height;
    }

    return matrix_builder;
}

CSR CSR::diagonally_dominant(unsigned n, double density, std::mt19937 rng) {
    assert(density * n >= 1 && "Matrix needs at least 1 element per row");

    constexpr unsigned lower = 1;
    constexpr unsigned upper = 10;
    constexpr unsigned l = 10;
    static_assert(lower < upper, "");

    std::uniform_int_distribution<> index_distrib(0, n - 1);
    std::uniform_real_distribution<> value_distrib(lower, upper);
    // TODO: more variety in the diagonal?
    std::uniform_real_distribution<> diag_distrib(density * n * upper, density * n * upper + l);
    // min value of max diagonal element to guarantee a big Eigenvalue
    // (follows from Gerschgorin circles): (3 * n - 2) * upper + l + 1;
    const double max_val = (3 * density * n - 2) * upper + l + 1;
    //const double max_val = 6 * n * upper + 10 * l;

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
        auto diag_index = std::distance(std::begin(row_colidx),
                                        std::find(std::begin(row_colidx), std::end(row_colidx), row));
        row_values.at(diag_index) = diag_distrib(rng);

        // update matrix
        values.insert(std::end(values), std::begin(row_values), std::end(row_values));
        colidx.insert(std::end(colidx), std::begin(row_colidx), std::end(row_colidx));
        rowptr.push_back(values.size());
    }
    // Put a value at position (0, 0) that guarantees an Eigenvalue >> than the other EVs
    // This leads to good convergence of the power iteration
    values[0] = max_val;

    return CSR{values, colidx, rowptr, n};
}

CSR CSR::diagonally_dominant_slice(unsigned n, double density, std::mt19937 &rng,
                                   unsigned first_row, unsigned last_row) {
    assert(density * n >= 1 && "CSR::diagonally_dominant_slice Matrix needs at least 1 element per row");
    assert(first_row <= last_row && "CSR::diagonally_dominant_slice First row <= last_row required");
    assert(last_row < n && "CSR::diagonally_dominant_slice last_row must still be in valid range");

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
    for (unsigned row = first_row; row <= last_row; ++row) {
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
        auto diag_index = std::distance(std::begin(row_colidx),
                                        std::find(std::begin(row_colidx), std::end(row_colidx), row));
        row_values.at(diag_index) = diag_distrib(rng);

        // update matrix
        values.insert(std::end(values), std::begin(row_values), std::end(row_values));
        colidx.insert(std::end(colidx), std::begin(row_colidx), std::end(row_colidx));
        rowptr.push_back(values.size());
    }

    return CSR{values, colidx, rowptr, n};
}

CSR CSR::fixed_eta(unsigned n, double density, double eta, std::mt19937 &rng) {
    assert(density >= 0. && density <= 1. && "CSR::fixed_eta Density needs to be in interval (0, 1)");
    assert(density * n >= 1 && "CSR::fixed_eta Matrix needs at least 1 element per row");
    assert(eta > 0. && eta < 1. && "CSR::fixed_eta eta needs to be in interval (0, 1)");

    constexpr double lambda_1 = 10.0;
    const double lambda_2 = lambda_1 * eta;

    std::uniform_int_distribution<> index_distrib(0, n - 1);

    // Generate two distinct but random rows to house the biggest two Eigenvalues
    unsigned lambda_1_row = index_distrib(rng);
    unsigned lambda_2_row = index_distrib(rng);
    while (lambda_1_row == lambda_2_row) {
        lambda_2_row = index_distrib(rng);
    }

    // The range for Eigenvalues has equal distance to lambda_2 and zero. Both won't be hit by Gerschgorin circles.
    constexpr double ev_fac = 0.75; // Determines the fraction of
    static_assert(ev_fac > 0.5 && ev_fac < 1,
                  "CSR::fixed_eta Eigenvalues that are neither lambda 1 nor lambda 2 must be in the range (0.5, 1) * lambda_2");
    const double ev_upper = lambda_2 * ev_fac;
    const double ev_lower = lambda_2 * (1. - ev_fac);
    std::uniform_real_distribution<> ev_distribution(ev_upper, ev_lower);

    // determine value-range for non-diagonal elements, s.t. their Gerschgorin circles do not contain lambda 2 or lambda 1, no matter what
    const auto num_non_diagonal = static_cast<unsigned>(density * n - 1);
    // 0.9 for a slightly bigger gap between Gerschgorin circles and lambda_2
    const double max_non_diag = 0.9 * (lambda_2 - ev_upper) / num_non_diagonal;
    std::uniform_real_distribution<> value_distribution(max_non_diag / 256., max_non_diag);

    std::vector<double> values;
    std::vector<int> colidx;
    std::vector<int> rowptr;
    rowptr.push_back(0);
    // construct all the rows
    for (unsigned row = 0; row < n; ++row) {
        if (row == lambda_1_row) {
            values.push_back(lambda_1);
            colidx.push_back(row);
        } else if (row == lambda_2_row) {
            values.push_back(lambda_2);
            colidx.push_back(row);
        } else { // Create a 'normal' matrix row where we do not particularly care about the Eigenvalues
            std::vector<double> row_values{};
            // values for 1 row, insert 1 extra to later replace with an eigenvalue
            for (unsigned k = 0; k < density * n; ++k) {
                row_values.push_back(value_distribution(rng));
            }

            std::set<int> colidx_set{};
            colidx_set.insert(row);
            while (colidx_set.size() < density * n) {
                colidx_set.insert(index_distrib(rng));
            }
            std::vector<int> row_colidx(colidx_set.size());
            std::copy(std::begin(colidx_set), std::end(colidx_set), std::begin(row_colidx));

            // insert diagonally dominant element
            auto diag_index = std::distance(std::begin(row_colidx),
                                            std::find(std::begin(row_colidx), std::end(row_colidx), row));
            row_values.at(diag_index) = ev_distribution(rng);

            // update matrix
            values.insert(std::end(values), std::begin(row_values), std::end(row_values));
            colidx.insert(std::end(colidx), std::begin(row_colidx), std::end(row_colidx));
            // rowptr.push_back(values.size());
        }
        rowptr.push_back(values.size());
    }

    return CSR{values, colidx, rowptr, n};
}

CSR CSR::random(uint64_t width, uint64_t height, double density, std::mt19937 rng) {
    assert(width > 0 && "CSR::random Matrix width needs to be > 1");
    assert(height > 0 && "CSR::random Matrix height needs to be > 1");
    assert(density >= 0 && density <= 1 && "CSR::random matrix density needs to be in [0, 1]");

    std::vector<std::vector<double>> value_matrix{height, std::vector<double>(width, 0.)};

    std::uniform_int_distribution<> x_distrib(0, width - 1);
    std::uniform_int_distribution<> y_distrib(0, height - 1);
    std::uniform_real_distribution<> value_distrib(0.001, 100'000.);

    unsigned num_values{0};
    while (width * (height * density) > num_values) {
        auto x = x_distrib(rng);
        auto y = y_distrib(rng);
        if (value_matrix.at(y).at(x) != 0) {
            continue;
        }
        auto val = value_distrib(rng);
        value_matrix.at(y).at(x) = val;
        ++num_values;
    }

    std::vector<double> values;
    std::vector<int> colidx;
    std::vector<int> rowptr{0};
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

void CSR::print() const {
    for (unsigned row{0}; row < this->num_rows(); ++row) {
        // print an empty row, then resume with the next row
        if (this->m_rowptr[row] == this->m_rowptr[row + 1]) {
            for (unsigned col{0}; col < this->num_cols(); ++col) {
                std::cout << "_ ";
            }
            std::cout << "\n";
            continue;
        }

        const auto row_start = this->m_rowptr[row];

        auto colidx_offset{0};
        unsigned curr_colidx = this->m_colidx.at(row_start);

        for (unsigned col{0}; col < this->num_cols(); ++col) {
            if (col == curr_colidx) {
                const int idx = row_start + colidx_offset;
                std::cout << this->m_values.at(idx) << " ";

                // we got everything from this row already, no need to update, also updating is dangerous!
                if (idx + 1 < this->m_rowptr.at(row + 1)) {
                    curr_colidx = this->m_colidx.at(idx + 1);
                    colidx_offset += 1;
                }
            } else {
                std::cout << "_ ";
            }
        }
        std::cout << "\n";
    }
}