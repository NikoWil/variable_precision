//
// Created by niko on 8/20/19.
//

#include <cstring>
#include "communication.h"

void bcast_vector(std::vector<double> &v, MPI_Comm comm, int root) {
    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    int size;
    if (rank == root) {
        size = v.size();
    }
    MPI_Bcast(&size, 1, MPI_INT, root, comm);

    if (rank != root) {
        v.clear();
        v.resize(size);
    }

    MPI_Bcast(v.data(), size, MPI_DOUBLE, root, comm);
}

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

CSR distribute_matrix(const CSR &matrix, MPI_Comm comm, int root) {
    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    const int metadata_size = 3;

    std::vector<int> r_num_rows;
    std::vector<int> r_num_values;
    std::vector<int> r_stretched_rowptr;
    std::vector<unsigned> r_metadata;

    std::vector<int> r_rowptr_displs;
    std::vector<int> r_values_displs;

    if (rank == root) {
        r_num_rows.resize(comm_size);
        r_num_values.resize(comm_size);
        for (int i = 0; i < comm_size; ++i) {
            unsigned start = (matrix.num_rows() * i) / comm_size;
            unsigned end = (matrix.num_rows() * (i + 1)) / comm_size;

            r_num_rows.at(i) = end - start;
            r_num_values.at(i) = matrix.rowptr().at(end) - matrix.rowptr().at(start);
        }

        // compute prefix sums
        r_rowptr_displs.resize(comm_size);
        r_rowptr_displs.at(0) = 0;
        std::partial_sum(r_num_rows.begin(), r_num_rows.end() - 1, r_rowptr_displs.begin() + 1);

        r_values_displs.resize(comm_size);
        r_values_displs.at(0) = 0;
        std::partial_sum(r_num_values.begin(), r_num_values.end() - 1, r_values_displs.begin() + 1);

        // The rowptr of a CSR matrix with n rows contains n+1 elements
        for (auto &e : r_num_rows) {
            e++;
        }

        // Duplicate indices, as MPI Scatterv does not guarantee correct behaviour
        // in case of overlapping send buffers
        r_stretched_rowptr = duplicate_indices(matrix.rowptr(), r_rowptr_displs);

        for (size_t i = 0; i < r_rowptr_displs.size(); ++i) {
            r_rowptr_displs.at(i) += i;
        }

        r_metadata.resize(comm_size * metadata_size);
        for (int i = 0; i < comm_size; ++i) {
            auto start = i * metadata_size;
            r_metadata.at(start + 0) = matrix.num_cols();
            r_metadata.at(start + 1) = r_num_rows.at(i);
            r_metadata.at(start + 2) = r_num_values.at(i);
        }
    }

    std::vector<unsigned> metadata(metadata_size);
    MPI_Scatter(r_metadata.data(), 3, MPI_UNSIGNED, metadata.data(), 3, MPI_UNSIGNED, root, comm);

    std::vector<int> rowptr(metadata.at(1));
    std::vector<int> colidx(metadata.at(2));
    std::vector<double> values(metadata.at(2));

    MPI_Scatterv(r_stretched_rowptr.data(), r_num_rows.data(), r_rowptr_displs.data(), MPI_INT, rowptr.data(),
                 rowptr.size(), MPI_INT, root, comm);
    MPI_Scatterv(matrix.colidx().data(), r_num_values.data(), r_values_displs.data(), MPI_INT, colidx.data(),
                 metadata.at(2), MPI_INT, root, comm);
    MPI_Scatterv(matrix.values().data(), r_num_values.data(), r_values_displs.data(), MPI_DOUBLE, values.data(),
                 metadata.at(2), MPI_DOUBLE, root, comm);

    // A node might own no row, if #nodes > #rows
    if (!rowptr.empty()) {
        int displacement = rowptr.at(0);

        for (auto &e : rowptr) {
            e -= displacement;
        }
    }

    return CSR{values, colidx, rowptr, metadata.at(0)};
}

/*void
gather_results(char *new_partial, char *old_partial, std::int32_t num_values_partial, std::int32_t num_values_total,
               std::int32_t bytes_per_val, char *out,
               std::int32_t out_bytes, const std::vector<std::int32_t> &rowcnt, MPI_Comm comm) {
    std::int32_t comm_size, rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);

    // ##########################################################
    // ### Compute & distribute the node specific information ###
    // ##########################################################
    std::vector<std::int32_t> relevant_per_rank(comm_size);

    std::int32_t needed_bytes{0};
    for (std::int32_t i{0}; i < num_values_partial * bytes_per_val; ++i) {
        if (new_partial[i] != old_partial[i]) {
            const std::int32_t byte_idx = i % bytes_per_val; // Index within the currently looked at value
            const std::int32_t affected_bytes =
                    bytes_per_val - byte_idx; // Look from the back end, as we assume little endian
            needed_bytes = std::max(needed_bytes, affected_bytes);
        }
    }
    MPI_Allgather(&needed_bytes, 1, MPI_INT32_T, relevant_per_rank.data(), 1, MPI_INT32_T, comm);

    // ##########################################################
    // ### Generate the data to distribute ######################
    // ##########################################################
    std::vector<char> sendbuf(num_values_partial * needed_bytes);
    for (size_t i{0}; i < num_values_partial; ++i) {
        const auto end = (i + 1) * bytes_per_val;
        const auto start = end - needed_bytes;
        std::memcpy(new_partial + start, sendbuf.data() + needed_bytes * i, needed_bytes);
    }

    // ##########################################################
    // ### Generate Allgatherv metadata on each rank ############
    // ##########################################################
    const std::int32_t sendcount{needed_bytes * num_values_partial};
    std::int32_t total_recvcount{0};
    for (std::int32_t i{0}; i < comm_size; ++i) {
        total_recvcount += rowcnt.at(i) * relevant_per_rank.at(i);
    }
    std::vector<char> recvbuf(total_recvcount);

    std::vector<std::int32_t> recvcounts;
    std::vector<std::int32_t> displs;
    recvcounts.reserve(comm_size);
    displs.reserve(comm_size);
    for (std::int32_t i{0}; i < comm_size; ++i) {
        recvcounts.push_back(relevant_per_rank.at(i) * rowcnt.at(i));
        displs.push_back(displs.back() + recvcounts.back());
    }

    MPI_Allgatherv(sendbuf.data(), sendcount, MPI_CHAR, recvbuf.data(), recvcounts.data(), displs.data(), MPI_CHAR,
                   comm);

    // ##########################################################
    // ### Update the result buffer #############################
    // ##########################################################
    std::int32_t out_start_idx{0};
    for (std::int32_t i{0}; i < comm_size; ++i) {
        const auto curr_relevant_bytes{relevant_per_rank.at(i)};
        const auto changed_start_idx{displs.at(i)}; // first byte belonging to the changed data from node i

        for (std::int32_t k{0}; k < rowcnt.at(i); ++k) {
            const auto changed_start{changed_start_idx + k * curr_relevant_bytes};
            const auto out_start{out_start_idx + (k + 1) * bytes_per_val - curr_relevant_bytes}; // change index to fit little endian, i.e. we only want to overwrite the curr_relevant_bytes last bytes of a value

            std::memcpy(recvbuf.data() + changed_start, out + out_start, curr_relevant_bytes);
        }

        out_start_idx += rowcnt.at(i) * bytes_per_val;
    }
}*/