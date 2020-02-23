//
// Created by Niko on 13/07/2019.
//

#ifndef CODE_UTIL_HPP
#define CODE_UTIL_HPP

#include <iostream>
#include <mpi.h>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
void print_vector(const std::vector<T> v, const std::string& name) {
    std::cout << name << ":\t";
    for (const auto& e : v) {
        std::cout << e << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void print_per_rank(T s, MPI_Comm comm) {
  int rank, comm_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  for (int k = 0; k < comm_size; ++k) {
    if (rank == k) {
      std::cout << "rank: " << rank << ", " <<  s << std::endl;
    }
    MPI_Barrier(comm);
  }
}

template <typename T>
std::vector<T> duplicate_indices(const std::vector<T>& values, const std::vector<int>& indices) {
    std::vector<T> buffed_values(values.size() + indices.size() - 1);

    unsigned int displs_index = 1;
    for (unsigned int i = 0; i < values.size(); i++) {
        buffed_values.at(i + displs_index - 1) = values.at(i);
        if (displs_index < (indices.size()) && i == static_cast<unsigned>(indices.at(displs_index))) {
            displs_index++;
            buffed_values.at(i + displs_index - 1) = values.at(i);
        }
    }

    return buffed_values;
}

template<typename T>
T median(std::vector<T> &v) {
    const auto size = v.size();
    const auto index_1 = size / 2;
    const auto index_2 = (size - 1) / 2;

    std::nth_element(v.begin(), v.begin() + index_1, v.end());
    const T e1 = v[index_1];
    std::nth_element(v.begin(), v.begin() + index_2, v.end());
    const T e2 = v[index_2];

    return (e1 + e2) / 2;
}

template<typename T>
T average(const std::vector<T> &v) {
    const auto sum = std::accumulate(v.begin(), v.end(), 0);
    return sum / v.size();
}

#endif // CODE_UTIL_HPP
