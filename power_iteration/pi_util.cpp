//
// Created by khondar on 18.03.20.
//

#include <assert.h>
#include <cmath>
#include "pi_util.h"

double dot(const std::vector<double> &x, const std::vector<double> &y) {
    assert(x.size() == y.size());
    double sum{0.};
    for (size_t i{0}; i < x.size(); ++i) {
        sum += x.at(i) * y.at(i);
    }
    return sum;
}

double norm(const std::vector<double> &v) { return sqrt(dot(v, v)); }

bool normalize(std::vector<double> &v) {
    const auto norm_fac = norm(v);
    if (norm_fac == 0) {
        return false;
    }

    for (size_t i{0}; i < v.size(); ++i) {
        v.at(i) = v.at(i) / norm_fac;
    }
    return true;
}

std::vector<double> scalar(const std::vector<double> &v, double s) {
    std::vector<double> scaled(v.size());
    for (size_t i{0}; i < v.size(); ++i) {
        scaled.at(i) = v.at(i) * s;
    }
    return scaled;
}