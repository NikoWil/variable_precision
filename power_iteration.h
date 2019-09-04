//
// Created by niko on 8/14/19.
//

#ifndef CODE_POWER_ITERATION_H
#define CODE_POWER_ITERATION_H

#include <mpi.h>
#include <vector>

#include "matrix_formats/csr.hpp"

std::pair<std::vector<double>, bool> power_iteration(const CSR& matrix, const std::vector<double>& x);

std::tuple<std::vector<double>, int, unsigned, bool> power_iteration(const CSR&matrix_slice, const std::vector<double>& x, const std::vector<int>& rowcnt, MPI_Comm comm);

std::tuple<std::vector<double>, unsigned, bool> power_iteration_fixed(const CSR&matrix_slice, const std::vector<double>&x, const std::vector<int>& rowcnt, MPI_Comm comm);

#endif // CODE_POWER_ITERATION_H
