//
// Created by niko on 8/20/19.
//

#ifndef CODE_COMMUNICATION_H
#define CODE_COMMUNICATION_H

#include <bitset>
#include <cmath>
#include <mpi.h>

#include "matrix_formats/csr.hpp"

void bcast_vector(std::vector<double> &v, MPI_Comm comm, int root);

CSR distribute_matrix(const CSR &matrix, MPI_Comm comm, int root);

void gather_results(char *new_partial, char *old_partial, size_t num_values, std::uint32_t bytes_per_val, char *out,
                    size_t out_bytes, const std::vector<int> &rowcnt, MPI_Comm comm);

#endif // CODE_COMMUNICATION_H
