//
// Created by niko on 6/6/19.
//
#include <iostream>

#include "csr.hpp"

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

CSR CSR::diagonally_dominant_slice(unsigned n, double density, std::mt19937& rng,
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
