#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "pi_benchmarks.h"
#include "spmv_benchmark.h"
#include "matrix_formats/csr.hpp"
#include "power_iteration/pagerank.h"
#include "segmentation/seg_uint.h"
#include "spmv/spmv_fixed.h"
#include "communication.h"
#include "spmv/pr_spmv.h"
#include "pagerank_test.h"

void get_rowcnt_start_row(MPI_Comm comm, unsigned num_rows, std::vector<int> &rowcnt, std::vector<int> &start_row) {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);

    rowcnt.clear();
    start_row.clear();
    start_row.push_back(0);

    for (int i{0}; i < comm_size; ++i) {
        unsigned start = (num_rows * i) / comm_size;
        unsigned end = (num_rows * (i + 1)) / comm_size;
        rowcnt.push_back(end - start);

        const auto last_start = start_row.back();
        start_row.push_back(last_start + rowcnt.back());
    }
}

void compare_partial_pr_spmv() {
    const unsigned num_cols = 5;

    const std::vector<double> values{0.5, 0.5, 1, 1, 1, 1};
    const std::vector<int> colidx{1, 4, 2, 0, 2, 2};
    const std::vector<int> rowptr{0, 2, 3, 4, 5, 6};
    CSR matrix{values, colidx, rowptr, num_cols};
    matrix = CSR::transpose(matrix);

    const std::vector<double> v1{1.};
    const std::vector<int> c1{2};
    const std::vector<int> r1{0, 1};
    const CSR m1{v1, c1, r1, num_cols};

    const std::vector<double> v2{0.5};
    const std::vector<int> c2{0};
    const std::vector<int> r2{0, 1};
    const CSR m2{v2, c2, r2, num_cols};

    const std::vector<double> v3{1., 1., 1.};
    const std::vector<int> c3{1, 3, 4};
    const std::vector<int> r3{0, 3};
    const CSR m3{v3, c3, r3, num_cols};

    const std::vector<double> v4{0.5};
    const std::vector<int> c4{0};
    const std::vector<int> r4{0, 0, 1};
    const CSR m4{v4, c4, r4, num_cols};

    std::vector<double> initial(num_cols, 1.);
    std::vector<double> result(matrix.num_rows());
    std::vector<double> result1(m1.num_rows());
    std::vector<double> result2(m2.num_rows());
    std::vector<double> result3(m3.num_rows());
    std::vector<double> result4(m4.num_rows());

    const double c{0.85};
    pagerank::fixed::spmv(matrix, initial, result, c);
    pagerank::fixed::spmv(m1, initial, result1, c);
    pagerank::fixed::spmv(m2, initial, result2, c);
    pagerank::fixed::spmv(m3, initial, result3, c);
    pagerank::fixed::spmv(m4, initial, result4, c);

    print_vector(result, "result");
    print_vector(result1, "result1");
    print_vector(result2, "result2");
    print_vector(result3, "result3");
    print_vector(result4, "result4");
}

int main(int argc, char *argv[]) {
    const auto requested = MPI_THREAD_FUNNELED;
    int provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    if (provided < requested) {
        std::cout << "No sufficient MPI multithreading support found\n";
        return 0;
    }

    const auto comm = MPI_COMM_WORLD;

    int rank, comm_size;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    test_convergence();

    // benchmark_spmv(20);

    MPI_Finalize();
    return 0;
}
