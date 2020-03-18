//
// Created by khondar on 18.03.20.
//

#ifndef CODE_PI_UTIL_H
#define CODE_PI_UTIL_H

#include <vector>

double dot(const std::vector<double> &x, const std::vector<double> &y);

double norm(const std::vector<double> &v);

bool normalize(std::vector<double> &v);

std::vector<double> scalar(const std::vector<double> &v, double s);

#endif //CODE_PI_UTIL_H
