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
    explicit CSR(const std::vector<double> &values, const std::vector<int> &colidx, const std::vector<int> &rowptr,
                 size_t num_cols)
            : m_values(values), m_colidx(colidx), m_rowptr(rowptr), m_num_cols(num_cols) {
        assert(values.size() == colidx.size() && "CSR size of values and colidx must be same");
        for (const auto &e : colidx) {
            (void) e;
            assert(e >= 0 && static_cast<unsigned>(e) < num_cols && "CSR each colidx must fulfill 0 <= idx < num_cols");
        }
        for (const auto &e : rowptr) {

            (void) e;
            assert(e >= 0 && static_cast<size_t>(e) <= values.size() &&
                   "CSR rowptr entries must fulfill 0 <= e <= values.size()");
        }

        assert(std::is_sorted(rowptr.begin(), rowptr.end()) && "CSR rowptr must be sorted in ascending order");
        assert(static_cast<size_t>(rowptr.back()) == values.size() &&
               "CSR rowptr last element must point at element after colidx/ values");
    }

    CSR() : m_values{}, m_colidx{}, m_rowptr{0}, m_num_cols{0} {}

    void concat_vertical(const CSR &other);

    void concat_horizontal(const CSR &other);

    static CSR transpose(const CSR &matrix);

    static CSR empty(std::size_t width, std::size_t height);

    static CSR unit(unsigned n);

    static CSR row_stochastic(unsigned n, double density, std::mt19937 rng);

    static CSR
    distributed_column_stochastic(std::size_t n, double density, std::mt19937 rng, std::size_t num_steps, MPI_Comm comm,
                                  int root = 0);

    static CSR diagonally_dominant(unsigned n, double density, std::mt19937 rng);

    static CSR diagonally_dominant_slice(unsigned n, double density,
                                         std::mt19937 &rng, unsigned first_row, unsigned last_row);

    static CSR fixed_eta(unsigned n, double density, double eta, std::mt19937 &rng);

    static CSR random(uint64_t width, uint64_t height, double density, std::mt19937 rng);

    const std::vector<double> &values() const {
        return m_values;
    }

    const std::vector<int> &colidx() const {
        return m_colidx;
    }

    const std::vector<int> &rowptr() const {
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

    void print() const;/* {
        print_vector(m_values, "m_values");
        print_vector(m_colidx, "m_colidx");
        print_vector(m_rowptr, "m_rowptr");
        std::cout << "m_num_cols: " << m_num_cols << std::endl;
        std::cout << "m_num_rows: " << this->num_rows() << std::endl;
    }// */

private:
    // this function takes a std::mt19937 by non-const reference to allow reproducible matrix generation
    // between this version and other row_stochastic matrices by using one std::mt19937 through multiple calls
    static CSR row_stochastic(unsigned width, unsigned height, double density, std::mt19937 &rng);

    std::vector<double> m_values;
    std::vector<int> m_colidx;
    std::vector<int> m_rowptr;
    size_t m_num_cols;
};

#endif // CODE_CSR_HPP
