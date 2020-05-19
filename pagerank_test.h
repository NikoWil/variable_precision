//
// Created by khondar on 14.04.20.
//

#ifndef CODE_PAGERANK_TEST_H
#define CODE_PAGERANK_TEST_H

#include <vector>
#include <mpi.h>

void test_convergence();

void test_precision_levels(unsigned n, double density, const std::vector<int> &rowcnt);

void pr_performance_test(MPI_Comm comm);

#endif //CODE_PAGERANK_TEST_H
