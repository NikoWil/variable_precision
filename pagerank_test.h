//
// Created by khondar on 14.04.20.
//

#ifndef CODE_PAGERANK_TEST_H
#define CODE_PAGERANK_TEST_H

#include <vector>

void test_convergence();

void test_precision_levels(unsigned n, double density, const std::vector<int> &rowcnt);

#endif //CODE_PAGERANK_TEST_H
