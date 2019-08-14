//
// Created by niko on 8/14/19.
//

#ifndef CODE_POWER_ITERATION_H
#define CODE_POWER_ITERATION_H

#include <mpi.h>
#include <vector>

#include "matrix_formats/csr.hpp"

std::vector<double> power_iteration(const CSR& matrix, const std::vector<double>& x, const std::vector<int>& rowcnt, MPI_Comm comm, int root);

#endif // CODE_POWER_ITERATION_H
