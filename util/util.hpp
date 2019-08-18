//
// Created by Niko on 13/07/2019.
//

#ifndef CODE_UTIL_HPP
#define CODE_UTIL_HPP

#include <iostream>
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

#endif // CODE_UTIL_HPP
