#include <cmath>
#include <iomanip>
#include <mpi.h>
#include <vector>

#include "communication.h"
#include "matrix_formats/csr.hpp"
#include "power_iteration.h"
#include "segmentation/segmentation.h"
#include "util/util.hpp"

int main(int argc, char* argv[]) {
  const auto requested = MPI_THREAD_FUNNELED;
  int provided;
  MPI_Init_thread(&argc, &argv, requested, &provided);
  if (provided < requested) {
    std::cout << "No sufficient MPI multithreading support found\n";
    return 0;
  }

  srand(static_cast<unsigned>(time(nullptr)));

  std::cout << std::setprecision(17);

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  std::vector<double> values;
  for (unsigned i = 0; i < 10; ++i) {
    values.push_back(rand() % 20);
  }
  std::vector<int> colidx{0, 1, 1, 2, 2, 3, 3, 4, 0, 4};
  std::vector<int> rowptr{0, 2, 4, 6, 8, 10};
  unsigned num_cols = 5;
  CSR matrix{values, colidx, rowptr, num_cols};

  std::vector<double> x;
  for (int i = 0; i < 5; ++i) {
    x.push_back(rand() % 20);
  }

  // Distribute & Print Matrix
  auto matrix_slice = distribute_matrix(matrix, MPI_COMM_WORLD, 0);
  for (int i = 0; i < comm_size; ++i) {
    if (i == rank) {
      std::cout << "rank: " << rank << std::endl;
      matrix_slice.print();
      std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::vector<int> rowcnt;
  for (int i = 0; i < comm_size; ++i) {
    unsigned start = (matrix.num_rows() * i) / comm_size;
    unsigned end = (matrix.num_rows() * (i + 1)) / comm_size;

    rowcnt.push_back(end - start);
  }

  auto result = power_iteration(matrix_slice, x, rowcnt, MPI_COMM_WORLD);
  if (rank == 0) {
    print_vector(result, "result");
  }

  auto result_simple = power_iteration(matrix, x);
  if (rank == 0) {
    print_vector(result_simple, "simple power iteration");
  }

  MPI_Finalize();

  return 0;
}