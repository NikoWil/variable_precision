//
// Created by khondar on 17.03.20.
//

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

std::pair<bool, int>
pagerank::fixed::pagerank(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                          MPI_Comm comm, const std::vector<int> &rowcnt, int iteration_limit) {
    /* preparation: asserts, get comm_size, comm_rank, etc
             *  get rank, comm_size
             *  calculate epsilon for precision (4 Bytes)
             *  calculate recvdispls -> prefix sum of rowcnt
             *
             *  NOT HERE (currently full precision power it)
             *    transform initial guess x to uint32_t
             *
             *  asserts:
             *    rowcnt[rank] == matrix.num_rows()
             *    x.size() == matrix.cols()
             *    result.size() == sum(rowcnt)
             */
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
    /*
     * Do Power Iteration
     *
     * Progress iteration:
     *  Local SpMV
     *    seg_uint::pre_convert::spmv_4(matrix, x, ...)
     *    -> local slice of A * y_k = z_{k+1}
     *
     *  Distribute results (MPI_Allgatherv)
     *    -> local copy of full z_{k+1}
     *    int MPI_Allgatherv(const void *sendbuf, int sendcount,
     *        MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
     *        const int displs[], MPI_Datatype recvtype, MPI_Comm comm)
     *
     *    MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_UINT32_T,
     *                   new_result.data(), rowcnt.data(), recvdispls.data(), MPI_UINT32_T, comm);
     *
     * Check stop conditions: => do asynchronously or not?
     *  calculate rayleigh quotient  (needs un_normalized z_{k+1})
     *    -> rho_k = y_k^H * z_{k+1} = y_k^H * A * y_k
     *  normalize result
     *    -> y_{k+1} = 1 / ||z_{k+1}|| * z_{k+1}
     *
     *  Potentially calculate asynchronously with values from previous iteration, while Allgatherv is taking place calculate norm-diff
     *    -> norm_diff = | y_k - y_{k+1} |
     *  calculate residual
     *    -> || y_{k+1} * rho_k - y_{k+1} ||
     *
     *  stop depending on norm_diff, residual
     */
    const double epsilon = pow(2, -52) * 10;

    std::vector<double> curr = initial;
    bool initial_non_zero = normalize<1>(curr);
    if (!initial_non_zero) {
        return std::make_pair(false, 0);
    }

    std::vector<double> next(initial.size());
    std::vector<double> partial_result(rowcnt.at(rank));

    bool done = false;
    int i{0};
    while (!done && i < iteration_limit) {
        // Calculate z_{k+1} = A * y_k
        //print_vector(curr, "curr");
        pagerank::fixed::spmv(matrix, curr, partial_result, c);
        MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_DOUBLE, next.data(), rowcnt.data(),
                       recvdispls.data(), MPI_DOUBLE, comm);
        //print_vector(next, "next");

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

std::pair<bool, int> pagerank::seg::pagerank_2(const CSR &matrix, const std::vector<std::uint16_t> &initial,
                                               std::vector<std::uint16_t> &result, double c, MPI_Comm comm,
                                               const std::vector<int> &rowcnt, int iteration_limit) {
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
        return std::make_pair(false, 0);
    }
    result.clear();
    result.resize(initial.size());
    std::vector<uint16_t> partial_result(rowcnt.at(rank));

    bool done = false;
    int i{0};
    while (!done && i < iteration_limit) {
        seg::spmv_2(matrix, curr, partial_result, c);
        MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_UINT16_T, result.data(),
                       rowcnt.data(), recvdispls.data(), MPI_UINT16_T, comm);

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
        done = norm_diff < epsilon; // && residual < epsilon;

        std::swap(result, curr);
        ++i;
    }

    std::swap(curr, result);
    return std::make_pair(done, i);
}

std::pair<bool, int> pagerank::seg::pagerank_4(const CSR &matrix, const std::vector<std::uint32_t> &initial,
                                               std::vector<std::uint32_t> &result, double c, MPI_Comm comm,
                                               const std::vector<int> &rowcnt, int iteration_limit) {
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
        return std::make_pair(false, 0);
    }
    result.clear();
    result.resize(initial.size());

    std::vector<std::uint32_t> partial_result(rowcnt.at(rank));

    bool done = false;
    int i{0};
    while (!done && i < iteration_limit) {
        seg::spmv_4(matrix, curr, partial_result, c);
        MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_UINT32_T, result.data(),
                       rowcnt.data(), recvdispls.data(), MPI_UINT32_T, comm);

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

        done = norm_diff < epsilon; // && residual < epsilon;

        std::swap(result, curr);
        ++i;
    }

    std::swap(curr, result);
    return std::make_pair(done, i);
}

std::pair<bool, int> pagerank::seg::pagerank_6(const CSR &matrix, const std::vector<std::uint16_t> &initial,
                                               std::vector<std::uint16_t> &result, double c, MPI_Comm comm,
                                               const std::vector<int> &rowcnt, int iteration_limit) {
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
        return std::make_pair(false, 0);
    }
    result.clear();
    result.resize(initial.size());

    std::vector<std::uint16_t> partial_result(rowcnt.at(rank) * 3);
    bool done = false;
    int i{0};
    while (!done && i < iteration_limit) {
        /*std::cout << "curr:\n\t";
        for (size_t k{0}; k < initial.size(); k += 3) {
            double val;
            seg_uint::read_6(&curr.at(k), &val);
            std::cout << val << " ";
        }
        std::cout << "\n";//*/

        seg::spmv_6(matrix, curr, partial_result, c);
        MPI_Allgatherv(partial_result.data(), 3 * rowcnt.at(rank), MPI_UINT16_T, result.data(),
                       recvcounts.data(), recvdispls.data(), MPI_UINT16_T, comm);

        /*std::cout << "partial_result:\n\t";
        for (size_t k{0}; k < partial_result.size(); k += 3) {
            double val;
            seg_uint::read_6(&partial_result.at(k), &val);
            std::cout << val << " ";
        }
        std::cout << "\n";

        std::cout << "result:\n\t";
        for (size_t k{0}; k < result.size(); k += 3) {
            double val;
            seg_uint::read_6(&result.at(k), &val);
            std::cout << val << " ";
        }
        std::cout << "\n";//*/

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

        done = norm_diff < epsilon;

        std::swap(result, curr);
        ++i;
    }

    std::swap(curr, result);
    return std::make_pair(done, i);
}

std::array<std::pair<bool, int>, 4>
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

    left_iterations -= meta_2.second;

    std::vector<std::uint32_t> initial_4(n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_2(&result_2.at(i), &val);
        seg_uint::write_4(&initial_4.at(i), &val);
    }
    std::vector<std::uint32_t> result_4(n);
    const auto meta_4 = seg::pagerank_4(matrix, initial_4, result_4, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_4.second;

    std::vector<std::uint16_t> initial_6(3 * n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_4(&result_4.at(i), &val);
        seg_uint::write_6(&initial_6.at(3 * i), &val);
    }
    std::vector<std::uint16_t> result_6(3 * n);
    const auto meta_6 = seg::pagerank_6(matrix, initial_6, result_6, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_6.second;

    std::vector<double> initial_8(n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_6(&result_6.at(3 * i), &val);
        initial_8.at(i) = val;
    }
    const auto meta_8 = fixed::pagerank(matrix, initial_8, result, c, comm, rowcnt, left_iterations);

    return {meta_2, meta_4, meta_6, meta_8};
}

std::array<std::pair<bool, int>, 3>
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

    left_iterations -= meta_4.second;

    std::vector<std::uint16_t> initial_6(3 * n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_4(&result_4.at(i), &val);
        seg_uint::write_6(&initial_6.at(3 * i), &val);
    }
    std::vector<std::uint16_t> result_6(3 * n);
    const auto meta_6 = seg::pagerank_6(matrix, initial_6, result_6, c, comm, rowcnt, left_iterations);

    left_iterations -= meta_6.second;

    std::vector<double> initial_8(n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_6(&result_6.at(3 * i), &val);
        initial_8.at(i) = val;
    }
    const auto meta_8 = fixed::pagerank(matrix, initial_8, result, c, comm, rowcnt, left_iterations);

    return {meta_4, meta_6, meta_8};
}

std::array<std::pair<bool, int>, 2>
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

    left_iterations -= meta_6.second;

    std::vector<double> initial_8(n);
    for (std::size_t i{0}; i < n; ++i) {
        double val;
        seg_uint::read_6(&result_6.at(3 * i), &val);
        initial_8.at(i) = val;
    }
    const auto meta_8 = fixed::pagerank(matrix, initial_8, result, c, comm, rowcnt, left_iterations);

    return {meta_6, meta_8};
}