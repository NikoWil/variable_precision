//
// Created by khondar on 08.02.20.
//

#ifndef CODE_PI_BENCHMARKS_H
#define CODE_PI_BENCHMARKS_H

#include <vector>
#include <mpi.h>

void
iteration_counter(const std::vector<int> &sizes, const std::vector<double> &densities, const std::vector<double> &etas,
                  unsigned max_iterations, int num_tests);

void
speedup_test(int size, double density, double eta, int max_iterations, int num_tests, const std::vector<int> &rowcnt,
             std::vector<unsigned> &seg_timings, std::vector<unsigned> &fixed_timings, MPI_Comm comm);

#endif //CODE_PI_BENCHMARKS_H
