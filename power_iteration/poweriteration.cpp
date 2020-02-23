//
// Created by niko on 11/14/19.
//

#include <limits>

#include "../spmv/spmv_fixed.h"
#include "../segmentation/seg_uint.h"
#include "poweriteration.h"

namespace {
    double dot(const std::vector<double> &x, const std::vector<double> &y) {
        assert(x.size() == y.size());
        double sum{0.};
        for (size_t i{0}; i < x.size(); ++i) {
            sum += x.at(i) * y.at(i);
        }
        return sum;
    }

    double norm(const std::vector<double> &v) { return sqrt(dot(v, v)); }

    bool normalize(std::vector<double> &v) {
        const auto norm_fac = norm(v);
        if (norm_fac == 0) {
            return false;
        }

        for (size_t i{0}; i < v.size(); ++i) {
            v.at(i) = v.at(i) / norm_fac;
        }
        return true;
    }

    std::vector<double> scalar(const std::vector<double> &v, double s) {
        std::vector<double> scaled(v.size());
        for (size_t i{0}; i < v.size(); ++i) {
            scaled.at(i) = v.at(i) * s;
        }
        return scaled;
    }

    /*std::vector<double> minus(const std::vector<double> &x,
                              const std::vector<double> &y) {
        assert(x.size() == y.size());

        std::vector<double> diff(x.size());
        for (size_t i{0}; i < x.size(); ++i) {
            diff.at(i) = x.at(i) - y.at(i);
        }
        return diff;
    }*/
}

std::pair<bool, int> local::power_iteration(const CSR &matrix, const std::vector<double> &x,
                                            std::vector<double> &curr, int iteration_limit) {
    curr = x;
    std::vector<double> next(x.size(), 0.);

    double curr_norm_diff{std::numeric_limits<double>::infinity()};
    double next_norm_diff{0};

    // Ignore the last 3.something digits of precision
    const double epsilon = pow(2, -52) * 10;

    bool done = false;
    int i{0};
    while (!done && i < iteration_limit) {
        // Calculate z_{k+1} = A * y_k
        fixed::spmv(matrix, curr, next);

        // Calculate Rayleigh-Quotient as y_k^H * z_{k+1}
        const double rayleigh = dot(curr, next);
        const auto next_copy = next;

        // Normalize our vector
        const bool normalized = normalize(next);
        if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
            break;
        }

        // Calculate residual
        const auto ev_vector = scalar(next, rayleigh);

        // Check for finish condition
        next_norm_diff = 0;
        for (size_t k{0}; k < next.size(); ++k) {
            next_norm_diff += std::abs(next[k] - curr[k]);
        }

        // Done if: we are suddenly taking bigger steps OR we are close enough
        // TODO: is 'suddenly taking bigger steps' an okay condition?
        //  Nope. It's not. How to improve it?
        //  -->> Residual?
        //done = next_norm_diff > curr_norm_diff;
        /*if (next_norm_diff > curr_norm_diff) {
          std::cout << "normdiff increasing! Curr: " << curr_norm_diff << ", next: " << next_norm_diff << "\n";
        }*/
        curr_norm_diff = next_norm_diff;

        done = done || curr_norm_diff < epsilon;

        std::swap(next, curr);
        ++i;
    }
    // std::cout << "Simple power iteration " << i << " iterations" << std::endl;

    return std::make_pair(done, i);
}

namespace distributed {
    namespace fixed {
        std::pair<bool, int> power_iteration(const CSR &matrix,
                                             const std::vector<double> &initial,
                                             std::vector<double> &result, MPI_Comm comm,
                                             int iteration_limit) {
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
                ::fixed::spmv(matrix, curr, partial_result);
                // TODO: distribution!
                MPI_Allgatherv(partial_result.data(), rowcnt.at(rank), MPI_DOUBLE,
                               next.data(), rowcnt.data(), recvdispls.data(), MPI_DOUBLE,
                               comm);

                // Calculate Rayleigh-Quotient as rho_k = y_k^H * z_{k+1} = y_k^H * A * y_k
                //const double rayleigh = dot(curr, next);
                // Save z_{k+1} for
                const auto next_copy = next;

                // Normalize z_{k+1} to get y_{k+1}
                const bool normalized = normalize(next);
                if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
                    std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
                    break;
                }

                // TODO: change to use y_k as would be correct? Do we still have that? Yes. In curr Calculate residual  || y_{k} * rho_k - A * y_{k} || = || y_{k} * rho_k - z_{k+1} ||
                // const auto ev_vector = scalar(curr, rayleigh);
                // const double residual = norm(minus(ev_vector, next_copy));

                // Check for finish condition
                double norm_diff = 0;
                for (size_t k{0}; k < next.size(); ++k) {
                    norm_diff += std::abs(next[k] - curr[k]);
                }

                /*if (rank == 0) {
                  std::cout << "Iteration " << i << "\n";
                  std::cout << "\tRayleigh-Quotient: " << rayleigh << "\n";
                  std::cout << "\tResidual: " << residual << "\n";
                  std::cout << "\tNormDiff: " << norm_diff << "\n\n";
                  // done = done || next_norm_diff < epsilon;
                  if (norm_diff < epsilon) {
                    std::cout << "next norm diff < epsilon!\n";
                  }
                  if (residual < epsilon) {
                    std::cout << "residual < epsilon!\n";
                  }
                }*/
                done = norm_diff < epsilon;

                std::swap(next, curr);
                ++i;
            }
            // std::cout << "Simple power iteration " << i << " iterations" << std::endl;

            std::swap(curr, result);
            return std::make_pair(done, i);
        }
    }

    namespace seg_uint {
        std::pair<bool, int>
        power_iteration_2(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
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

            assert(static_cast<unsigned>(rowcnt.at(rank)) == matrix.num_rows());
            assert(initial.size() == matrix.num_cols());

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
            const double epsilon = pow(2, -4) * 10;

            std::vector<double> curr = initial;
            bool initial_non_zero = normalize(curr);
            if (!initial_non_zero) {
                return std::make_pair(false, 0);
            }

            std::vector<double> next(initial.size());
            std::vector<uint16_t> next_halves(initial.size());

            std::vector<double> partial_result(rowcnt.at(rank));
            std::vector<uint16_t> partial_result_halves(rowcnt.at(rank));

            bool done = false;
            int i{0};
            while (!done && i < iteration_limit) {
                // Calculate z_{k+1} = A * y_k
                ::seg_uint::out_convert::spmv_2(matrix, curr, partial_result_halves);

                MPI_Allgatherv(partial_result_halves.data(), rowcnt.at(rank), MPI_UINT16_T, next_halves.data(),
                               rowcnt.data(), recvdispls.data(), MPI_UINT16_T, comm);
                for (size_t k{0}; k < next_halves.size(); ++k) {
                    ::seg_uint::read_2(&next_halves.at(k), &next.at(k));
                }

                // Calculate Rayleigh-Quotient as rho_k = y_k^H * z_{k+1} = y_k^H * A * y_k
                //const double rayleigh = dot(curr, next);
                // Save z_{k+1} for residual
                const auto next_copy = next;
                // Normalize z_{k+1} to get y_{k+1}
                const bool normalized = normalize(next);
                if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
                    std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
                    break;
                }

                // Check for finish condition
                //const auto ev_vector = scalar(curr, rayleigh);
                //const double residual = norm(minus(ev_vector, next_copy));
                double norm_diff = 0;
                for (size_t k{0}; k < next.size(); ++k) {
                    norm_diff += std::abs(next[k] - curr[k]);
                }

                /*if (rank == 0) {
                  std::cout << "Iteration " << i << "\n";
                  std::cout << "\tRayleigh-Quotient: " << rayleigh << "\n";
                  std::cout << "\tResidual: " << residual << "\n";
                  std::cout << "\tNormDiff: " << norm_diff << "\n\n";
                  // done = done || next_norm_diff < epsilon;
                  if (norm_diff < epsilon) {
                    std::cout << "next norm diff < epsilon!\n";
                  }
                  if (residual < epsilon) {
                    std::cout << "residual < epsilon!\n";
                  }
                }*/
                done = norm_diff < epsilon; // && residual < epsilon;

                std::swap(next, curr);
                ++i;
            }
            // std::cout << "Simple power iteration " << i << " iterations" << std::endl;

            std::swap(curr, result);
            return std::make_pair(done, i);
        }

        std::pair<bool, int>
        power_iteration_4(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
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
            const double epsilon = pow(2, -20) * 10;

            std::vector<double> curr = initial;
            bool initial_non_zero = normalize(curr);
            if (!initial_non_zero) {
                return std::make_pair(false, 0);
            }

            std::vector<double> next(initial.size());
            std::vector<uint32_t> next_halves(initial.size());

            std::vector<uint32_t> partial_result_halves(rowcnt.at(rank));

            bool done = false;
            int i{0};
            while (!done && i < iteration_limit) {
                // Calculate z_{k+1} = A * y_k
                ::seg_uint::out_convert::spmv_4(matrix, curr, partial_result_halves);

                MPI_Allgatherv(partial_result_halves.data(), rowcnt.at(rank), MPI_UINT32_T,
                               next_halves.data(), rowcnt.data(), recvdispls.data(), MPI_UINT32_T,
                               comm);
                for (size_t k{0}; k < next_halves.size(); ++k) {
                    ::seg_uint::read_4(&next_halves.at(k), &next.at(k));
                }

                // Calculate Rayleigh-Quotient as rho_k = y_k^H * z_{k+1} = y_k^H * A * y_k
                //const double rayleigh = dot(curr, next);
                // Save z_{k+1} for residual
                const auto next_copy = next;
                // Normalize z_{k+1} to get y_{k+1}
                const bool normalized = normalize(next);
                if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
                    std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
                    break;
                }

                // Check for finish condition
                //const auto ev_vector = scalar(curr, rayleigh);
                //const double residual = norm(minus(ev_vector, next_copy));
                double norm_diff = 0;
                for (size_t k{0}; k < next.size(); ++k) {
                    norm_diff += std::abs(next[k] - curr[k]);
                }

                /*if (rank == 0) {
                  std::cout << "Iteration " << i << "\n";
                  std::cout << "\tRayleigh-Quotient: " << rayleigh << "\n";
                  std::cout << "\tResidual: " << residual << "\n";
                  std::cout << "\tNormDiff: " << norm_diff << "\n\n";
                  // done = done || next_norm_diff < epsilon;
                  if (norm_diff < epsilon) {
                    std::cout << "next norm diff < epsilon!\n";
                  }
                  if (residual < epsilon) {
                    std::cout << "residual < epsilon!\n";
                  }
                }*/
                done = norm_diff < epsilon; // && residual < epsilon;

                std::swap(next, curr);
                ++i;
            }
            // std::cout << "Simple power iteration " << i << " iterations" << std::endl;

            std::swap(curr, result);
            return std::make_pair(done, i);
        }

        std::pair<bool, int>
        power_iteration_6(const CSR &matrix, const std::vector<double> &initial, std::vector<double> &result,
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
                const auto new_displs = recvdispls.back() + 3 * rowcnt.at(i);
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
            const double epsilon = pow(2, -36) * 10;

            std::vector<double> curr = initial;
            bool initial_non_zero = normalize(curr);
            if (!initial_non_zero) {
                return std::make_pair(false, 0);
            }

            std::vector<int> recvcounts;
            for (const auto r : rowcnt) {
                recvcounts.push_back(3 * r);
            }

            std::vector<double> next(initial.size());
            std::vector<uint16_t> next_halves(3 * initial.size());

            std::vector<uint16_t> partial_result_halves(3 * rowcnt.at(rank));
            bool done = false;
            int i{0};
            while (!done && i < iteration_limit) {
                // Calculate z_{k+1} = A * y_k
                ::seg_uint::out_convert::spmv_6(matrix, curr, partial_result_halves);

                MPI_Allgatherv(partial_result_halves.data(), 3 * rowcnt.at(rank), MPI_UINT16_T,
                               next_halves.data(), recvcounts.data(), recvdispls.data(), MPI_UINT16_T,
                               comm);
                for (size_t k{0}; k < next.size(); ++k) {
                    ::seg_uint::read_6(&next_halves.at(3 * k), &next.at(k));
                }

                // Calculate Rayleigh-Quotient as rho_k = y_k^H * z_{k+1} = y_k^H * A * y_k
                //const double rayleigh = dot(curr, next);
                // Save z_{k+1} for residual
                const auto next_copy = next;
                // Normalize z_{k+1} to get y_{k+1}
                const bool normalized = normalize(next);
                if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
                    std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
                    break;
                }

                // Check for finish condition
                //const auto ev_vector = scalar(curr, rayleigh);
                //const double residual = norm(minus(ev_vector, next_copy));
                double norm_diff = 0;
                for (size_t k{0}; k < next.size(); ++k) {
                    norm_diff += std::abs(next[k] - curr[k]);
                }

                /*if (rank == 0) {
                  std::cout << "Iteration " << i << "\n";
                  std::cout << "\tRayleigh-Quotient: " << rayleigh << "\n";
                  std::cout << "\tResidual: " << residual << "\n";
                  std::cout << "\tNormDiff: " << norm_diff << "\n\n";
                  // done = done || next_norm_diff < epsilon;
                  if (norm_diff < epsilon) {
                    std::cout << "next norm diff < epsilon!\n";
                  }
                  if (residual < epsilon) {
                    std::cout << "residual < epsilon!\n";
                  }
                }*/
                done = norm_diff < epsilon; // && residual < epsilon;

                std::swap(next, curr);
                ++i;
            }
            // std::cout << "Simple power iteration " << i << " iterations" << std::endl;

            std::swap(curr, result);
            return std::make_pair(done, i);
        }
    }
}
