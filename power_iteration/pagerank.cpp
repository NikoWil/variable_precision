//
// Created by khondar on 17.03.20.
//

#include "pagerank.h"
#include "pi_util.h"
#include "../spmv/pr_spmv.h"

std::pair<bool, int>
pagerank::fixed::pagerank(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result, double c,
                          MPI_Comm comm, int iteration_limit) {
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

    const auto local_rowcnt = matrix.num_rows();
    std::vector<int> rowcnt(comm_size);
    MPI_Allgather(&local_rowcnt, 1, MPI_INT, rowcnt.data(), 1, MPI_INT, comm);

    std::vector<int> recvdispls;
    recvdispls.push_back(0);
    for (size_t i{0}; i < rowcnt.size() - 1; ++i) {
        const auto new_displs = recvdispls.back() + rowcnt.at(i);
        recvdispls.push_back(new_displs);
    }

    unsigned rowsum{0};
    for (size_t i{0}; i < rowcnt.size(); ++i) {
        rowsum += rowcnt.at(i);
    }
    // Asserts:
    assert(static_cast<unsigned>(rowcnt.at(rank)) == matrix.num_rows());
    assert(initial.size() == matrix.num_cols());
    assert(rowsum == matrix.num_cols());

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
    bool initial_non_zero = normalize(curr);
    if (!initial_non_zero) {
        return std::make_pair(false, 0);
    }

    std::vector<double> next(initial.size());
    std::vector<double> partial_result(rowcnt.at(rank));

    bool done = false;
    int i{0};
    while (!done && i < iteration_limit) {
        // Calculate z_{k+1} = A * y_k
        spmv(matrix, curr, partial_result, c);
        MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_DOUBLE, next.data(), rowcnt.data(),
                       recvdispls.data(), MPI_DOUBLE, comm);

        // Normalize z_{k+1} to get y_{k+1}
        const bool normalized = normalize(next);
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
