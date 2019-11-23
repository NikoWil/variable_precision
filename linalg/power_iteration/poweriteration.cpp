//
// Created by niko on 11/14/19.
//

#include <limits>

#include "../spmv/spmv_fixed.h"
#include "poweriteration.h"

double dot(const std::vector<double>& x, const std::vector<double>& y) {
  assert(x.size() == y.size());
  double sum{0.};
  for (size_t i{0}; i < x.size(); ++i) {
    sum += x.at(i) * y.at(i);
  }
  return sum;
}

double norm(const std::vector<double>& v) {
  return sqrt(dot(v, v));
}

bool normalize(std::vector<double>& v) {
  const auto norm_fac = norm(v);
  if (norm_fac == 0) {
    return false;
  }

  for (size_t i{0}; i < v.size(); ++i) {
    v.at(i) = v.at(i) / norm_fac;
  }
  return true;
}

std::vector<double> scalar(const std::vector<double>& v, double s) {
  std::vector<double> scaled(v.size());
  for (size_t i{0}; i < v.size(); ++i) {
    scaled.at(i) = v.at(i) * s;
  }
  return scaled;
}

std::vector<double> minus(const std::vector<double>& x, const std::vector<double>& y) {
  assert(x.size() == y.size());

  std::vector<double> diff(x.size());
  for (size_t i{0}; i < x.size(); ++i) {
    diff.at(i) = x.at(i) - y.at(i);
  }
  return diff;
}

std::pair<bool, int> local::power_iteration(const CSR &matrix, const std::vector<double> &x,
                           std::vector<double>&curr, int iteration_limit) {
  curr = x;
  std::vector<double> next(x.size(), 0.);

  double curr_norm_diff{std::numeric_limits<double>::infinity()};
  double next_norm_diff{0};

  double rayleigh{0};

  // Ignore the last 3.something digits of precision
  const double epsilon = pow(2, -52) * 10;

  bool done = false;
  int i{0};
  while (!done && i < iteration_limit) {
    std::cout << "Iteration " << i << "\n";
    // Calculate z_{k+1} = A * y_k
    fixed::spmv(matrix, curr, next);

    // Calculate Rayleigh-Quotient as y_k^H * z_{k+1}
    rayleigh = dot(curr, next);
    std::cout << "\tRayleigh-Quotient: " << rayleigh << "\n";
    const auto next_copy = next;

    // Normlize our vector
    const bool normalized = normalize(next);
    if (!normalized) { // NaN NaN NaN NaN NaN NaN BATMAN!
      std::cout << "NaN NaN NaN NaN NaN NaN BATMAN!\n";
      break;
    }

    // Calculate residual
    const auto ev_vector = scalar(next, rayleigh);
    const double residual = norm(minus(ev_vector, next_copy));
    std::cout << "\tResidual: " << residual << "\n";

    // Check for finish condition
    next_norm_diff = 0;
    for (size_t k{0}; k < next.size(); ++k) {
      next_norm_diff += std::abs(next[k] - curr[k]);
    }
    std::cout << "\tNormDiff: " << next_norm_diff << "\n\n";
    // Done if: we are suddenly taking bigger steps OR we are close enough
    // TODO: is 'suddenly taking bigger steps' an okay condition?
    //  Nope. It's not. How to improve it?
    //  -->> Residual?
    //done = next_norm_diff > curr_norm_diff;
    if (next_norm_diff > curr_norm_diff) {
      std::cout << "normdiff increasing! Curr: " << curr_norm_diff << ", next: " << next_norm_diff << "\n";
    }
    curr_norm_diff = next_norm_diff;

    //done = done || next_norm_diff < epsilon;
    if (curr_norm_diff < epsilon) {
      std::cout << "next norm diff < epsilon!\n";
    }

    const bool same = next == curr;
    if (same) {
      std::cout << "curr and new curr are same! Normdiff: " << curr_norm_diff << "\n";
    }
    done = done || same;

    std::swap(next, curr);
    ++i;
  }
  // std::cout << "Simple power iteration " << i << " iterations" << std::endl;

  std::swap(next, curr);
  return std::make_pair(done, i);
}