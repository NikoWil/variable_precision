//
// Created by Niko on 13/07/2019.
//

#ifndef CODE_UTIL_HPP
#define CODE_UTIL_HPP

#include <iostream>
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

template <typename T>
std::vector<T> prefix_sums(const std::vector<T>& v) {
    std::vector<T> sums(v.size());
    sums.at(0) = 0;

    for (unsigned i = 0; i < v.size() - 1; i++) {
        sums.at(i + 1) = sums.at(i) + v.at(i);
    }

    return sums;
}

#endif // CODE_UTIL_HPP
