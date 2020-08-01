#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <chrono>

#include "matrix_formats/csr.hpp"
#include "power_iteration/pagerank.h"
#include "communication.h"
#include "spmv/pr_spmv.h"
#include "segmentation/seg_uint.h"
#include "pagerank_test.h"

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

void seg_pr_speed_test(int n, double density) {
    const auto comm = MPI_COMM_WORLD;

    int rank, comm_size;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    std::mt19937 rng{std::random_device{}()};

    const auto matrix = CSR::transpose(CSR::row_stochastic(n, density, rng));
    std::cout << "Setup done" << std::endl;
    const auto matrix_slice = distribute_matrix(matrix, comm, 0);
    std::cout << "Matrix distributed" << std::endl;
    std::vector<double> initial(n, 1.);
    std::vector<double> result(n, 0.);
    const double c = 0.85;

    std::vector<int> rowcnt, start_row;
    get_rowcnt_start_row(comm, n, rowcnt, start_row);

    const std::uint32_t warmup{50};
    const std::uint32_t num_tests{300};

    double sum{0};

    std::vector<std::array<pagerank::pr_meta, 4>> var_metas;
    var_metas.reserve(4 * num_tests);
    std::vector<pagerank::pr_meta> fix_metas;
    fix_metas.reserve(num_tests);

    for (std::uint32_t i{0}; i < warmup; ++i) {
        pagerank::fixed::pagerank(matrix_slice, initial, result, c, comm, rowcnt);
        sum += result[0];
        initial[0] = i + 1;
    }
    std::cout << "fixed warmup" << std::endl;
    for (std::uint32_t i{0}; i < num_tests; ++i) {
        auto meta = pagerank::fixed::pagerank(matrix_slice, initial, result, c, comm, rowcnt);
        fix_metas.push_back(std::move(meta));

        sum += result[0];
        initial[0] = i + 1.;
    }
    std::cout << "fixed done" << std::endl;

    initial[0] = 1.;
    for (std::uint32_t i{0}; i < warmup; ++i) {
        pagerank::variable::pagerank_2_4_6_8(matrix_slice, initial, result, c, comm, rowcnt);
        sum += result[0];
        initial[0] = i + 1;
    }

    for (std::uint32_t i{0}; i < num_tests; ++i) {
        auto meta_var = pagerank::variable::pagerank_2_4_6_8(matrix_slice, initial, result, c, comm, rowcnt);
        var_metas.push_back(std::move(meta_var));
        sum += result[0];
        initial[0] = i + 1.;
    }

    if (rank == 0) {
        std::vector<pagerank::pr_meta> var_metas_2;
        std::vector<pagerank::pr_meta> var_metas_4;
        std::vector<pagerank::pr_meta> var_metas_6;
        std::vector<pagerank::pr_meta> var_metas_8;
        for (const auto &m : var_metas) {
            var_metas_2.push_back(m[0]);
            var_metas_4.push_back(m[1]);
            var_metas_6.push_back(m[2]);
            var_metas_8.push_back(m[3]);
        }

        const auto extract_medians = [](const std::vector<pagerank::pr_meta> &metas) {
            std::vector<std::int64_t> prep_times;
            std::vector<std::int64_t> spmv_times;
            std::vector<std::int64_t> agv_times;
            std::vector<std::int64_t> ovhd_times;
            for (const auto &m : metas) {
                prep_times.push_back(m.prep_timing);
                spmv_times.insert(spmv_times.end(), m.spmv_timings.begin(), m.spmv_timings.end());
                agv_times.insert(agv_times.end(), m.agv_timings.begin(), m.agv_timings.end());
                ovhd_times.insert(ovhd_times.end(), m.overhead_timings.begin(), m.overhead_timings.end());
            }

            return std::tuple{median(prep_times), median(spmv_times), median(agv_times), median(ovhd_times)};
        };

        const auto extract_averages = [](const std::vector<pagerank::pr_meta> &metas) {
            std::vector<std::int64_t> prep_times;
            std::vector<std::int64_t> spmv_times;
            std::vector<std::int64_t> agv_times;
            std::vector<std::int64_t> ovhd_times;
            for (const auto &m : metas) {
                prep_times.push_back(m.prep_timing);
                spmv_times.insert(spmv_times.end(), m.spmv_timings.begin(), m.spmv_timings.end());
                agv_times.insert(agv_times.end(), m.agv_timings.begin(), m.agv_timings.end());
                ovhd_times.insert(ovhd_times.end(), m.overhead_timings.begin(), m.overhead_timings.end());
            }

            return std::tuple{average(prep_times), average(spmv_times), average(agv_times), average(ovhd_times)};
        };

        const auto[prep_2_med, spmv_2_med, agv_2_med, ovhd_2_med] = extract_medians(var_metas_2);
        const auto[prep_4_med, spmv_4_med, agv_4_med, ovhd_4_med] = extract_medians(var_metas_4);
        const auto[prep_6_med, spmv_6_med, agv_6_med, ovhd_6_med] = extract_medians(var_metas_6);
        const auto[prep_8_med, spmv_8_med, agv_8_med, ovhd_8_med] = extract_medians(var_metas_8);
        const auto[prep_fix_med, spmv_fix_med, agv_fix_med, ovhd_fix_med] = extract_medians(fix_metas);

        const auto[prep_2_avg, spmv_2_avg, agv_2_avg, ovhd_2_avg] = extract_averages(var_metas_2);
        const auto[prep_4_avg, spmv_4_avg, agv_4_avg, ovhd_4_avg] = extract_averages(var_metas_4);
        const auto[prep_6_avg, spmv_6_avg, agv_6_avg, ovhd_6_avg] = extract_averages(var_metas_6);
        const auto[prep_8_avg, spmv_8_avg, agv_8_avg, ovhd_8_avg] = extract_averages(var_metas_8);
        const auto[prep_fix_avg, spmv_fix_avg, agv_fix_avg, ovhd_fix_avg] = extract_averages(fix_metas);

        std::cout << "Average\t| preparation \tspmv \t\tallgatherv \toverhead\n";
        std::cout << "-------------------------------------------------------------------\n";
        std::cout << "2\t| " << prep_2_avg << "\t" << spmv_2_avg << "\t\t" << agv_2_avg << "\t\t" << ovhd_2_avg << "\n";
        std::cout << "4\t| " << prep_4_avg << "\t" << spmv_4_avg << "\t\t" << agv_4_avg << "\t\t" << ovhd_4_avg << "\n";
        std::cout << "6\t| " << prep_6_avg << "\t" << spmv_6_avg << "\t\t" << agv_6_avg << "\t\t" << ovhd_6_avg << "\n";
        std::cout << "8\t| " << prep_8_avg << "\t" << spmv_8_avg << "\t\t" << agv_8_avg << "\t\t" << ovhd_8_avg << "\n";
        std::cout << "fixed\t| " << prep_fix_avg << "\t" << spmv_fix_avg << "\t\t" << agv_fix_avg << "\t\t"
                  << ovhd_fix_avg << "\n";

        std::cout << "\n";

        std::cout << "Median\t| preparation \tspmv \t\tallgatherv \toverhead\n";
        std::cout << "-------------------------------------------------------------------\n";
        std::cout << "2\t| " << prep_2_med << "\t" << spmv_2_med << "\t\t" << agv_2_med << "\t\t" << ovhd_2_med << "\n";
        std::cout << "4\t| " << prep_4_med << "\t" << spmv_4_med << "\t\t" << agv_4_med << "\t\t" << ovhd_4_med << "\n";
        std::cout << "6\t| " << prep_6_med << "\t" << spmv_6_med << "\t\t" << agv_6_med << "\t\t" << ovhd_6_med << "\n";
        std::cout << "8\t| " << prep_8_med << "\t" << spmv_8_med << "\t\t" << agv_8_med << "\t\t" << ovhd_8_med << "\n";
        std::cout << "fxed\t| " << prep_fix_med << "\t" << spmv_fix_med << "\t\t" << agv_fix_med << "\t\t"
                  << ovhd_fix_med << "\n";
    }
}

void seg_speed_test(const int n) {
    std::vector<double> v1;
    v1.reserve(n);
    for (int i{0}; i < n; ++i) {
        v1.push_back(i);
    }
    std::vector<double> v2(n);
    std::reverse_copy(v1.begin(), v1.end(), v2.begin());

    std::vector<std::uint16_t> v1_2b;
    std::vector<std::uint32_t> v1_4b;
    std::vector<std::array<std::uint16_t, 3>> v1_6b;

    std::vector<std::uint16_t> v2_2b;
    std::vector<std::uint32_t> v2_4b;
    std::vector<std::array<std::uint16_t, 3>> v2_6b;
    for (int i{0}; i < n; ++i) {
        v1_2b.push_back(seg_uint::write_2(&v1.at(i)));
        v1_4b.push_back(seg_uint::write_4(&v1.at(i)));
        v1_6b.push_back(seg_uint::write_6(&v1.at(i)));

        v2_2b.push_back(seg_uint::write_2(&v2.at(i)));
        v2_4b.push_back(seg_uint::write_4(&v2.at(i)));
        v2_6b.push_back(seg_uint::write_6(&v2.at(i)));
    }

    std::vector<double> result(n);
    std::vector<std::uint16_t> result_2(n);
    std::vector<std::uint32_t> result_4(n);
    std::vector<std::array<std::uint16_t, 3>> result_6(n);

    using namespace std::chrono;
    const auto start = high_resolution_clock::now();
#pragma omp parallel for default(none) shared(v1, v2, result)
    for (int i = 0; i < n; ++i) {
        result.at(i) = v1.at(i) * v2.at(i);
    }
    const auto end = high_resolution_clock::now();

    const auto start_2 = high_resolution_clock::now();
#pragma omp parallel for default(none) shared(v1_2b, v2_2b, result_2)
    for (int i = 0; i < n; ++i) {
        const double d1 = seg_uint::read_2(&v1_2b.at(i));
        const double d2 = seg_uint::read_2(&v2_2b.at(i));
        const double mul = d1 * d2;
        result_2.at(i) = seg_uint::write_2(&mul);
    }
    const auto end_2 = high_resolution_clock::now();

    const auto start_4 = high_resolution_clock::now();
#pragma omp parallel for default(none) shared(v1_4b, v2_4b, result_4)
    for (int i = 0; i < n; ++i) {
        const double d1 = seg_uint::read_4(&v1_4b.at(i));
        const double d2 = seg_uint::read_4(&v2_4b.at(i));
        const double mul = d1 * d2;
        result_4.at(i) = seg_uint::write_4(&mul);
    }
    const auto end_4 = high_resolution_clock::now();

    const auto start_6 = high_resolution_clock::now();
#pragma omp parallel for default(none) shared(v1_6b, v2_6b, result_6)
    for (int i = 0; i < n; ++i) {
        const double d1 = seg_uint::read_6(v1_6b.at(i));
        const double d2 = seg_uint::read_6(v2_6b.at(i));
        const double mul = d1 * d2;
        result_6.at(i) = seg_uint::write_6(&mul);
    }
    const auto end_6 = high_resolution_clock::now();

    const auto total = duration_cast<milliseconds>(end - start).count();
    const auto total_2 = duration_cast<milliseconds>(end_2 - start_2).count();
    const auto total_4 = duration_cast<milliseconds>(end_4 - start_4).count();
    const auto total_6 = duration_cast<milliseconds>(end_6 - start_6).count();

    std::cout << "8 bytes: " << total << "\n";
    std::cout << "2 bytes: " << total_2 << "\n";
    std::cout << "4 bytes: " << total_4 << "\n";
    std::cout << "6 bytes: " << total_6 << "\n";

}

int main(int argc, char *argv[]) {
    const auto requested = MPI_THREAD_FUNNELED;
    int provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    if (provided < requested) {
        std::cout << "No sufficient MPI multithreading support found\n";
        return 0;
    }

    //pr_performance_test(comm);
    const int size = std::atoi(argv[1]);
    const double density = std::atof(argv[2]);

    /*const std::vector<double> values{1};
    const std::vector<int> colidx{0};
    const std::vector<int> rowptr{0, 1, 1, 1};
    const std::size_t num_cols{2};
    CSR matrix{values, colidx, rowptr, num_cols};//*/

    int comm_rank, comm_size;
    const auto comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);
    constexpr int root{0};

    if (comm_rank == root) {
        std::cout << "size: " << size << ", density: " << density << "\n";
    }
    const auto seed = std::random_device{}();
    const auto matrix_local = CSR::row_stochastic(size, 1. / size, std::mt19937{seed});

    const auto matrix_distrib = CSR::distributed_column_stochastic(size, 1. / size, std::mt19937{seed}, 2, comm, root);

    /*if (comm_rank == root) {
        std::cout << "matrix local\n";
        CSR::transpose(matrix_local).print();
    }*/

    for (int i{0}; i < comm_size; ++i) {
        if (comm_rank == i) {
            std::cout << "rank " << i << " matrix distrib\n";
            matrix_distrib.print();
        }
        MPI_Barrier(comm);
    }

    /*const double c{0.85};
    const unsigned warmup{1};
    const unsigned test_iterations{1};
    const auto comm = MPI_COMM_WORLD;

    std::vector<int> rowcnt, start_row;
    get_rowcnt_start_row(comm, size, rowcnt, start_row);

    single_speedup_test(size, density, c, warmup, test_iterations, comm, rowcnt);// */

    MPI_Finalize();
    return 0;
}
