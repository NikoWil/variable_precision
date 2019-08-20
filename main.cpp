#include <mpi.h>
#include <vector>

#include "communication.h"
#include "matrix_formats/csr.hpp"
#include "power_iteration.h"
#include "util/util.hpp"

void power_iteration_test() {
  std::vector<double> values;
  for (unsigned i = 0; i < 11; ++i) {
    values.push_back(rand());
  }

  std::vector<int>    colidx{0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5};
  std::vector<int>    rowptr{0,      2,      4,      6,      8,     10, 11};
  unsigned num_cols = 6;
  CSR matrix{values, colidx, rowptr, num_cols};

  std::vector<double> x;
  for (unsigned i = 0; i < num_cols; i++) {
    x.push_back(rand());
  }

  std::vector<int> rowcnt{6};

  const auto result = power_iteration(matrix, x, rowcnt, MPI_COMM_WORLD);
  print_vector(result, "power iteration result");
}

int main(int argc, char* argv[]) {
  const auto requested = MPI_THREAD_FUNNELED;
  int provided;
  MPI_Init_thread(&argc, &argv, requested, &provided);
  if (provided < requested) {
    std::cout << "No sufficient MPI multithreading support found\n";
    return 0;
  }

  srand(static_cast<unsigned>(time(0)));

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  std::vector<double> values{1., 2., 3., 4., 5.};
  std::vector<int> colidx{0, 1, 2, 3, 4};
  std::vector<int> rowptr{0, 1, 2, 3, 4, 5};
  unsigned num_cols = 5;
  CSR matrix{values, colidx, rowptr, num_cols};

  auto matrix_slice = distribute_matrix(matrix, MPI_COMM_WORLD, 0);
  for (int i = 0; i < comm_size; ++i) {
    if (i == rank) {
      std::cout << "rank: " << rank << "\n";
      matrix_slice.print();
      std::cout << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}