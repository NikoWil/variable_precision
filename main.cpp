#include <mpi.h>
#include <vector>

#include "matrix_formats/csr.hpp"

int main(int argc, char* argv[]) {
  const auto requested = MPI_THREAD_FUNNELED;
  int provided;
  MPI_Init_thread(&argc, &argv, requested, &provided);
  if (provided < requested) {
    std::cout << "No sufficient MPI multithreading support found\n";
    return 0;
  }

  int n = 30;
  CSR matrix = CSR::unit(n);
  std::vector<double> x(n, 1.);
  std::vector<double> result = matrix.spmv(x);

  MPI_Finalize();

  return 0;
}