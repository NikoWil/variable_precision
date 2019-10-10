//
// Created by niko on 8/20/19.
//

#ifndef CODE_COMMUNICATION_H
#define CODE_COMMUNICATION_H

#include <bitset>
#include <cmath>
#include <mpi.h>

#include "matrix_formats/csr.hpp"
#include "segmentation_char/segmentation_char.h"

void bcast_vector(std::vector<double>& v, MPI_Comm comm, int root);

CSR distribute_matrix(const CSR& matrix, MPI_Comm comm, int root);

#endif // CODE_COMMUNICATION_H
