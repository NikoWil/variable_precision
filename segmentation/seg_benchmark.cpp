//
// Created by niko on 11/13/19.
//
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <random>

#include "seg_char.h"
#include "seg_uint.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  std::mt19937 rng{std::random_device{}()};
  size_t size{1u << 26u};
  std::uniform_real_distribution<>distrib (0., 1u << 31u);

  std::vector<double> x(size);
  std::vector<uint16_t> seg_uint_2(x.size());
  std::vector<uint32_t> seg_uint_4(x.size());
  std::vector<uint16_t> seg_uint_6(x.size() * 3);
  std::vector<seg::Double_slice<0, 1>> seg_char_2(x.size());
  std::vector<seg::Double_slice<0, 3>> seg_char_4(x.size());
  std::vector<seg::Double_slice<0, 5>> seg_char_6(x.size());

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if (rank == 0) {
    for (size_t i{0}; i < x.size(); ++i) {
      x.at(i) = distrib(rng);
    }

    MPI_Send(x.data(), x.size(), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);

    MPI_Recv(seg_uint_2.data(), seg_uint_2.size(), MPI_UINT16_T, 1, 0, MPI_COMM_WORLD, nullptr);
    MPI_Recv(seg_uint_4.data(), seg_uint_2.size(), MPI_UINT32_T, 1, 1, MPI_COMM_WORLD, nullptr);
    MPI_Recv(seg_uint_6.data(), seg_uint_2.size(), MPI_UINT16_T, 1, 2, MPI_COMM_WORLD, nullptr);
    MPI_Recv(seg_char_2.data(), seg_char_2.size(), MPI_CHAR, 1, 3, MPI_COMM_WORLD, nullptr);
    MPI_Recv(seg_char_4.data(), seg_char_4.size(), MPI_CHAR, 1, 4, MPI_COMM_WORLD, nullptr);
    MPI_Recv(seg_char_6.data(), seg_char_6.size(), MPI_CHAR, 1, 5, MPI_COMM_WORLD, nullptr);
  } else if (rank == 1) {
    MPI_Recv(x.data(), x.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, nullptr);

    using namespace std::chrono;
    auto start_u2 = high_resolution_clock::now();
    for (size_t i{0}; i < x.size(); ++i) {
      seg_uint::write_2(seg_uint_2.data() + i, x.data() + i);
    }
    auto end_u2 = high_resolution_clock::now();
    std::cout << "seg_uint 2: " << duration_cast<milliseconds>(end_u2 - start_u2).count() << std::endl;

    auto start_u4 = high_resolution_clock::now();
    for (size_t i{0}; i < x.size(); ++i) {
      seg_uint::write_4(seg_uint_4.data() + i, x.data() + i);
    }
    auto end_u4 = high_resolution_clock::now();
    std::cout << "seg_uint 4: " << duration_cast<milliseconds>(end_u4 - start_u4).count() << std::endl;

    auto start_u6 = high_resolution_clock::now();
    for (size_t i{0}; i < x.size(); ++i) {
      seg_uint::write_6(seg_uint_6.data() + (i * 3), x.data() + i);
    }
    auto end_u6 = high_resolution_clock::now();
    std::cout << "seg_uint 6: " << duration_cast<milliseconds>(end_u6 - start_u6).count() << std::endl;

    auto start_c2 = high_resolution_clock::now();
    for (size_t i{0}; i < x.size(); ++i) {
      seg_char_2[i] = seg::Double_slice<0, 1>{x[i]};
    }
    auto end_c2 = high_resolution_clock::now();
    std::cout << "seg_char 2: " << duration_cast<milliseconds>(end_c2 - start_c2).count() << std::endl;

    auto start_c4 = high_resolution_clock::now();
    for (size_t i{0}; i < x.size(); ++i) {
      seg_char_4[i] = seg::Double_slice<0, 3>{x[i]};
    }
    auto end_c4 = high_resolution_clock::now();
    std::cout << "seg_char 4: " << duration_cast<milliseconds>(end_c4 - start_c4).count() << std::endl;

    auto start_c6 = high_resolution_clock::now();
    for (size_t i{0}; i < x.size(); ++i) {
      seg_char_6[i] = seg::Double_slice<0, 5>{x[i]};
    }
    auto end_c6 = high_resolution_clock::now();
    std::cout << "seg_char 6: " << duration_cast<milliseconds>(end_c6 - start_c6).count() << std::endl;

    MPI_Send(seg_uint_2.data(), seg_uint_2.size(), MPI_UINT16_T, 0, 0, MPI_COMM_WORLD);
    MPI_Send(seg_uint_4.data(), seg_uint_2.size(), MPI_UINT32_T, 0, 1, MPI_COMM_WORLD);
    MPI_Send(seg_uint_6.data(), seg_uint_2.size(), MPI_UINT16_T, 0, 2, MPI_COMM_WORLD);
    MPI_Send(seg_char_2.data(), seg_char_2.size(), MPI_CHAR, 0, 3, MPI_COMM_WORLD);
    MPI_Send(seg_char_4.data(), seg_char_4.size(), MPI_CHAR, 0, 4, MPI_COMM_WORLD);
    MPI_Send(seg_char_6.data(), seg_char_6.size(), MPI_CHAR, 0, 5, MPI_COMM_WORLD);
  } else {
    std::cout << "ERROR in seg_benchmark. More than 2 nodes existing! Rank " << rank << std::endl;
  }

  MPI_Finalize();
}