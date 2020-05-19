//
// Created by khondar on 17.03.20.
//

#include <chrono>
#include "pagerank.h"
#include "pi_util.h"
#include "../spmv/pr_spmv.h"

std::pair<bool, int>
pagerank::local::pagerank(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                          int iteration_limit) {
    const double epsilon = pow(2, -52) * 10;

    std::vector<double> curr = initial;
    bool initial_non_zero = normalize<1>(curr);
    if (!initial_non_zero) {
        return std::make_pair(false, 0);
    }

    std::vector<double> next(initial.size());

    bool done = false;
    int i{0};
    while (!done && i < iteration_limit) {
        // Calculate z_{k+1} = A * y_k
        pagerank::fixed::spmv(matrix, curr, next, c);

        // Normalize z_{k+1} to get y_{k+1}
        const bool normalized = normalize<1>(next);
        if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
            std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
            break;
        }

        // Check for finish condition
        double norm_diff = 0;
        for (size_t k{0}; k < next.size(); ++k) {
            norm_diff += std::abs(next[k] - curr[k]);
        }

        done = norm_diff < epsilon;

        std::swap(next, curr);
        ++i;
    }

    std::swap(curr, result);
    return std::make_pair(done, i);
}

pagerank::pr_meta
pagerank::fixed::pagerank(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                          MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit) {
    using namespace std::chrono;
    const auto total_start = high_resolution_clock::now();
    const auto prep_start = high_resolution_clock::now();

    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    std::vector<int> recvdispls;
    recvdispls.push_back(0);
    for (size_t i{0}; i < rowcnt.size() - 1; ++i) {
        const auto new_displs = recvdispls.back() + rowcnt.at(i);
        recvdispls.push_back(new_displs);
    }

    {
        unsigned rowsum{0};
        for (size_t i{0}; i < rowcnt.size(); ++i) {
            rowsum += rowcnt.at(i);
        }
        // Asserts:
        assert(static_cast<unsigned>(rowcnt.at(rank)) == matrix.num_rows());
        assert(initial.size() == matrix.num_cols());
        assert(rowsum == matrix.num_cols());
    }
    const double epsilon = pow(2, -52) * 10;

    std::vector<double> curr = initial;
    bool initial_non_zero = normalize<1>(curr);
    if (!initial_non_zero) {
        return {false, 0, 0, 0, std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{}};
    }

    std::vector<double> next(initial.size());
    std::vector<double> partial_result(rowcnt.at(rank));

    bool converged = false;
    int i{0};

    const auto prep_end = high_resolution_clock::now();
    std::vector<std::int64_t> spmv_timings;
    std::vector<std::int64_t> agv_timings;
    std::vector<std::int64_t> overhead_timings;
    while (!converged && i < iteration_limit) {
        // Calculate z_{k+1} = A * y_k
        const auto spmv_start = high_resolution_clock::now();
        pagerank::fixed::spmv(matrix, curr, partial_result, c);
        const auto spmv_end = high_resolution_clock::now();
        const auto agv_start = high_resolution_clock::now();
        MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_DOUBLE, next.data(), rowcnt.data(),
                       recvdispls.data(), MPI_DOUBLE, comm);
        const auto agv_end = high_resolution_clock::now();

        const auto overhead_start = high_resolution_clock::now();
        // Normalize z_{k+1} to get y_{k+1}
        const bool normalized = normalize<1>(next);
        if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
            std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
            break;
        }

        // Check for finish condition
        double norm_diff = 0;
        for (size_t k{0}; k < next.size(); ++k) {
            norm_diff += std::abs(next[k] - curr[k]);
        }

        converged = norm_diff < epsilon;

        std::swap(next, curr);
        ++i;
        const auto overhead_end = high_resolution_clock::now();

        // calculate all time measurements
        spmv_timings.push_back(duration_cast<nanoseconds>(spmv_end - spmv_start).count());
        agv_timings.push_back(duration_cast<nanoseconds>(agv_end - agv_start).count());
        overhead_timings.push_back(duration_cast<nanoseconds>(overhead_end - overhead_start).count());
    }
    std::swap(curr, result);

    const auto prep_time = duration_cast<nanoseconds>(prep_end - prep_start).count();
    const auto total_end = high_resolution_clock::now();
    const auto total_time = duration_cast<nanoseconds>(total_end - total_start).count();
    return {converged, i, total_time, prep_time, spmv_timings, agv_timings, overhead_timings};
}

pagerank::pr_meta pagerank::seg::pagerank_2(const CSR &matrix, const std::vector<std::uint16_t> &initial,
                                            std::vector<std::uint16_t> &result, double c, MPI_Comm comm,
                                            const std::vector<int> &rowcnt, int iteration_limit) {
    using namespace std::chrono;
    const auto total_start = high_resolution_clock::now();
    const auto prep_start = high_resolution_clock::now();

    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    assert(static_cast<unsigned>(rowcnt.at(rank)) == matrix.num_rows());
    assert(initial.size() == matrix.num_cols());

    std::vector<int> recvdispls;
    recvdispls.push_back(0);
    for (size_t i{0}; i < rowcnt.size() - 1; ++i) {
        const auto new_displs = recvdispls.back() + rowcnt.at(i);
        recvdispls.push_back(new_displs);
    }

    const double epsilon = pow(2, -4) * 10;

    /**
     * Create vectors to store calculations
     */
    std::vector<uint16_t> curr = initial;
    bool initial_non_zero = normalize_2<1>(curr);
    if (!initial_non_zero) {
        return {false, 0, 0, 0, std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{}};
    }
    result.clear();
    result.resize(initial.size());
    std::vector<uint16_t> partial_result(rowcnt.at(rank));

    bool converged = false;
    int i{0};

    const auto prep_end = high_resolution_clock::now();
    std::vector<std::int64_t> spmv_timings;
    std::vector<std::int64_t> agv_timings;
    std::vector<std::int64_t> overhead_timings;
    while (!converged && i < iteration_limit) {
        const auto spmv_start = high_resolution_clock::now();
        seg::spmv_2(matrix, curr, partial_result, c);
        const auto spmv_end = high_resolution_clock::now();
        const auto agv_start = high_resolution_clock::now();
        MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_UINT16_T, result.data(),
                       rowcnt.data(), recvdispls.data(), MPI_UINT16_T, comm);
        const auto agv_end = high_resolution_clock::now();

        const auto overhead_start = high_resolution_clock::now();
        // Normalize z_{k+1} to get y_{k+1}
        const bool normalized = normalize_2<1>(result);
        if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
            std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
            break;
        }
        // Check for finish condition
        double norm_diff = 0;
        for (size_t k{0}; k < result.size(); ++k) {
            double val_1, val_2;
            seg_uint::read_2(&result[k], &val_1);
            seg_uint::read_2(&curr[k], &val_2);
            norm_diff += std::abs(val_1 - val_2);
        }
        converged = norm_diff < epsilon; // && residual < epsilon;

        std::swap(result, curr);
        ++i;
        const auto overhead_end = high_resolution_clock::now();

        // calculate all time measurements
        spmv_timings.push_back(duration_cast<nanoseconds>(spmv_end - spmv_start).count());
        agv_timings.push_back(duration_cast<nanoseconds>(agv_end - agv_start).count());
        overhead_timings.push_back(duration_cast<nanoseconds>(overhead_end - overhead_start).count());
    }
    std::swap(curr, result);

    const auto prep_time = duration_cast<nanoseconds>(prep_end - prep_start).count();
    const auto total_end = high_resolution_clock::now();
    const auto total_time = duration_cast<nanoseconds>(total_end - total_start).count();
    return {converged, i, total_time, prep_time, spmv_timings, agv_timings, overhead_timings};
}

pagerank::pr_meta pagerank::seg::pagerank_4(const CSR &matrix, const std::vector<std::uint32_t> &initial,
                                            std::vector<std::uint32_t> &result, double c, MPI_Comm comm,
                                            const std::vector<int> &rowcnt, int iteration_limit) {
    using namespace std::chrono;
    const auto total_start = high_resolution_clock::now();
    const auto prep_start = high_resolution_clock::now();

    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    assert(static_cast<unsigned>(rowcnt.at(rank)) == matrix.num_rows());
    assert(initial.size() == matrix.num_cols());

    std::vector<int> recvdispls;
    recvdispls.push_back(0);
    for (size_t i{0}; i < rowcnt.size() - 1; ++i) {
        const auto new_displs = recvdispls.back() + rowcnt.at(i);
        recvdispls.push_back(new_displs);
    }

    const double epsilon = pow(2, -20) * 10;

    /**
     * Create vectors to store calculations
     */
    std::vector<std::uint32_t> curr = initial;
    bool initial_non_zero = normalize_4<1>(curr);
    if (!initial_non_zero) {
        return {false, 0, 0, 0, std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{}};
    }
    result.clear();
    result.resize(initial.size());

    std::vector<std::uint32_t> partial_result(rowcnt.at(rank));

    bool converged = false;
    int i{0};

    const auto prep_end = high_resolution_clock::now();
    std::vector<std::int64_t> spmv_timings;
    std::vector<std::int64_t> agv_timings;
    std::vector<std::int64_t> overhead_timings;
    while (!converged && i < iteration_limit) {
        const auto spmv_start = high_resolution_clock::now();
        seg::spmv_4(matrix, curr, partial_result, c);
        const auto spmv_end = high_resolution_clock::now();
        const auto agv_start = high_resolution_clock::now();
        MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_UINT32_T, result.data(),
                       rowcnt.data(), recvdispls.data(), MPI_UINT32_T, comm);
        const auto agv_end = high_resolution_clock::now();
        const auto overhead_start = high_resolution_clock::now();
        // Normalize z_{k+1} to get y_{k+1}
        const bool normalized = normalize_4<1>(result);
        if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
            std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
            break;
        }

        // Check for finish condition
        double norm_diff = 0;
        for (size_t k{0}; k < result.size(); ++k) {
            double val_1, val_2;
            seg_uint::read_4(&result[k], &val_1);
            seg_uint::read_4(&curr[k], &val_2);
            norm_diff += std::abs(val_1 - val_2);
        }

        converged = norm_diff < epsilon; // && residual < epsilon;

        std::swap(result, curr);
        ++i;
        const auto overhead_end = high_resolution_clock::now();

        // calculate all time measurements
        spmv_timings.push_back(duration_cast<nanoseconds>(spmv_end - spmv_start).count());
        agv_timings.push_back(duration_cast<nanoseconds>(agv_end - agv_start).count());
        overhead_timings.push_back(duration_cast<nanoseconds>(overhead_end - overhead_start).count());
    }
    std::swap(curr, result);

    const auto prep_time = duration_cast<nanoseconds>(prep_end - prep_start).count();
    const auto total_end = high_resolution_clock::now();
    const auto total_time = duration_cast<nanoseconds>(total_end - total_start).count();
    return {converged, i, total_time, prep_time, spmv_timings, agv_timings, overhead_timings};
}

pagerank::pr_meta pagerank::seg::pagerank_6(const CSR &matrix, const std::vector<std::uint16_t> &initial,
                                            std::vector<std::uint16_t> &result, double c, MPI_Comm comm,
                                            const std::vector<int> &rowcnt, int iteration_limit) {
    using namespace std::chrono;
    const auto total_start = high_resolution_clock::now();
    const auto prep_start = high_resolution_clock::now();

    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    assert(static_cast<unsigned>(rowcnt.at(rank)) == matrix.num_rows());
    assert(initial.size() == 3 * matrix.num_cols());

    std::vector<int> recvdispls;
    recvdispls.push_back(0);
    for (size_t i{0}; i < rowcnt.size() - 1; ++i) {
        const auto new_displs = recvdispls.back() + 3 * rowcnt.at(i);
        recvdispls.push_back(new_displs);
    }

    std::vector<int> recvcounts;
    for (const auto r : rowcnt) {
        recvcounts.push_back(3 * r);
    }

    const double epsilon = pow(2, -36) * 10;

    // Create vectors to store calculations
    std::vector<std::uint16_t> curr = initial;
    bool initial_non_zero = normalize_6<1>(curr);
    if (!initial_non_zero) {
        return {false, 0, 0, 0, std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{}};
    }
    result.clear();
    result.resize(initial.size());

    std::vector<std::uint16_t> partial_result(rowcnt.at(rank) * 3);
    bool converged = false;
    int i{0};

    const auto prep_end = high_resolution_clock::now();
    std::vector<std::int64_t> spmv_timings;
    std::vector<std::int64_t> agv_timings;
    std::vector<std::int64_t> overhead_timings;
    while (!converged && i < iteration_limit) {
        const auto spmv_start = high_resolution_clock::now();
        seg::spmv_6(matrix, curr, partial_result, c);
        const auto spmv_end = high_resolution_clock::now();
        const auto agv_start = high_resolution_clock::now();
        MPI_Allgatherv(partial_result.data(), 3 * rowcnt.at(rank), MPI_UINT16_T, result.data(),
                       recvcounts.data(), recvdispls.data(), MPI_UINT16_T, comm);
        const auto agv_end = high_resolution_clock::now();
        const auto overhead_start = high_resolution_clock::now();
        // Normalize z_{k+1} to get y_{k+1}
        const bool normalized = normalize_6<1>(result);
        if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
            std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
            break;
        }

        // Check for finish condition
        double norm_diff = 0;
        for (size_t k{0}; k < result.size(); k += 3) {
            double val_1, val_2;
            seg_uint::read_6(&result[k], &val_1);
            seg_uint::read_6(&curr[k], &val_2);
            norm_diff += std::abs(val_1 - val_2);
        }

        converged = norm_diff < epsilon;

        std::swap(result, curr);
        ++i;
        const auto overhead_end = high_resolution_clock::now();

        // calculate all time measurements
        spmv_timings.push_back(duration_cast<nanoseconds>(spmv_end - spmv_start).count());
        agv_timings.push_back(duration_cast<nanoseconds>(agv_end - agv_start).count());
        overhead_timings.push_back(duration_cast<nanoseconds>(overhead_end - overhead_start).count());
    }
    std::swap(curr, result);

    const auto prep_time = duration_cast<nanoseconds>(prep_end - prep_start).count();
    const auto total_end = high_resolution_clock::now();
    const auto total_time = duration_cast<nanoseconds>(total_end - total_start).count();
    return {converged, i, total_time, prep_time, spmv_timings, agv_timings, overhead_timings};
}

std::array<pagerank::pr_meta, 4>
pagerank::variable::pagerank_2_4_6_8(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
                                     double c, MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit) {
    const std::size_t n{result.size()};
    int left_iterations = iteration_limit;

    std::vector<std::uint16_t> initial_2(n);
    for (std::size_t i{0}; i < n; ++i) {
        seg_uint::write_2(&initial_2.at(i), &initial.at(i));
    }
    std::vector<std::uint16_t> result_2(n);
    const auto meta_2 = seg::pagerank_2(matrix, initial_2, result_2, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_2.used_iterations;

    std::vector<std::uint32_t> initial_4(n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_2(&result_2.at(i), &val);
        seg_uint::write_4(&initial_4.at(i), &val);
    }
    std::vector<std::uint32_t> result_4(n);
    const auto meta_4 = seg::pagerank_4(matrix, initial_4, result_4, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_4.used_iterations;

    std::vector<std::uint16_t> initial_6(3 * n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_4(&result_4.at(i), &val);
        seg_uint::write_6(&initial_6.at(3 * i), &val);
    }
    std::vector<std::uint16_t> result_6(3 * n);
    const auto meta_6 = seg::pagerank_6(matrix, initial_6, result_6, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_6.used_iterations;

    std::vector<double> initial_8(n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_6(&result_6.at(3 * i), &val);
        initial_8.at(i) = val;
    }
    const auto meta_8 = fixed::pagerank(matrix, initial_8, result, c, comm, rowcnt, left_iterations);

    return {meta_2, meta_4, meta_6, meta_8};
}

std::array<pagerank::pr_meta, 3>
pagerank::variable::pagerank_4_6_8(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
                                   double c, MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit) {
    const std::size_t n{result.size()};
    int left_iterations = iteration_limit;

    std::vector<std::uint32_t> initial_4(n);
    for (std::size_t i{0}; i < n; ++i) {
        seg_uint::write_4(&initial_4.at(i), &initial.at(i));
    }
    std::vector<std::uint32_t> result_4(n);
    const auto meta_4 = seg::pagerank_4(matrix, initial_4, result_4, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_4.used_iterations;

    std::vector<std::uint16_t> initial_6(3 * n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_4(&result_4.at(i), &val);
        seg_uint::write_6(&initial_6.at(3 * i), &val);
    }
    std::vector<std::uint16_t> result_6(3 * n);
    const auto meta_6 = seg::pagerank_6(matrix, initial_6, result_6, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_6.used_iterations;

    std::vector<double> initial_8(n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_6(&result_6.at(3 * i), &val);
        initial_8.at(i) = val;
    }
    const auto meta_8 = fixed::pagerank(matrix, initial_8, result, c, comm, rowcnt, left_iterations);

    return {meta_4, meta_6, meta_8};
}

std::array<pagerank::pr_meta, 2>
pagerank::variable::pagerank_6_8(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
                                 double c, MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit) {
    const std::size_t n{result.size()};
    int left_iterations = iteration_limit;

    std::vector<std::uint16_t> initial_6(3 * n);
    for (std::size_t i{0}; i < n; ++i) {
        seg_uint::write_6(&initial_6.at(3 * i), &initial.at(i));
    }
    std::vector<std::uint16_t> result_6(3 * n);
    const auto meta_6 = seg::pagerank_6(matrix, initial_6, result_6, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_6.used_iterations;

    std::vector<double> initial_8(n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_6(&result_6.at(3 * i), &val);
        initial_8.at(i) = val;
    }
    const auto meta_8 = fixed::pagerank(matrix, initial_8, result, c, comm, rowcnt, left_iterations);

    return {meta_6, meta_8};
}

void pagerank::print_meta(const pagerank::pr_meta &meta) {
    std::cout << "converged " << meta.converged << "\n"
              << "iterations " << meta.used_iterations << "\n"
              << "prep_timing " << meta.prep_timing << "\n";
    print_vector(meta.spmv_timings, "spmv_timings");
    print_vector(meta.agv_timings, "agv_timings");
    print_vector(meta.overhead_timings, "overhead_timings");
}

void pagerank::print_fixed(const pagerank::pr_meta &meta) {
    std::cout << "fixed\n";
    print_meta(meta);
    std::cout << "\n";
}

void pagerank::print_2_4_6_8(const std::array<pagerank::pr_meta, 4> &meta) {
    std::cout << "variable_2_4_6_8\n";
    std::cout << "2_byte_precision\n";
    print_meta(meta[0]);
    std::cout << "4_byte_precision\n";
    print_meta(meta[1]);
    std::cout << "6_byte_precision\n";
    print_meta(meta[2]);
    std::cout << "8_byte_precision\n";
    print_meta(meta[3]);
    std::cout << "\n";
}