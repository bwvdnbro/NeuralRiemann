#include "Matrix.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#define NSAMPLE 100

template <uint_fast32_t _M_, uint_fast32_t _N_>
inline static Matrix<_M_, _N_> sigmoid(const Matrix<_M_, _N_> &a) {
  Matrix<_M_, _N_> result;
  for (uint_fast32_t m = 0; m < _M_; ++m) {
    for (uint_fast32_t n = 0; n < _N_; ++n) {
      result(m, n) = 1. / (1. + std::exp(-a(m, n)));
    }
  }
  return result;
}

inline static double random_double() {
  return ((double)rand()) / ((double)RAND_MAX);
}

inline static double gaussian_random_double() {
  const double u1 = std::sqrt(-2. * std::log(random_double()));
  const double u2 = 2. * M_PI * random_double();
  return u1 * std::cos(u2);
}

template <uint_fast32_t _M_, uint_fast32_t _N_>
inline static Matrix<_M_, _N_> random_matrix() {
  Matrix<_M_, _N_> result;
  for (uint_fast32_t m = 0; m < _M_; ++m) {
    for (uint_fast32_t n = 0; n < _N_; ++n) {
      result(m, n) = random_double();
    }
  }
  return result;
}

template <uint_fast32_t _M_, uint_fast32_t _N_>
inline static Matrix<_M_, _N_> gaussian_random_matrix() {
  Matrix<_M_, _N_> result;
  for (uint_fast32_t m = 0; m < _M_; ++m) {
    for (uint_fast32_t n = 0; n < _N_; ++n) {
      result(m, n) = gaussian_random_double();
    }
  }
  return result;
}

int main(int argc, char **argv) {

  const double eta = 0.001;

  const Matrix<NSAMPLE, 6> x = random_matrix<NSAMPLE, 6>();
  Matrix<NSAMPLE, 5> y;
  std::ofstream reffile("reference.txt");
  for (uint_fast32_t i = 0; i < NSAMPLE; ++i) {
    const uint_fast8_t idx = x(i, 0) / 0.2;
    y(i, idx) = 1.;
    const double result =
        std::round(y(i, 1) + 2. * y(i, 2) + 3. * y(i, 3) + 4. * y(i, 4));
    reffile << x(i, 0) << " : " << result << "\n";
  }
  reffile.close();

  Matrix<6, 5> hidden1_w = gaussian_random_matrix<6, 5>();
  Matrix<1, 5> hidden1_b = gaussian_random_matrix<1, 5>();
  Matrix<5, 10> hidden2_w = gaussian_random_matrix<5, 10>();
  Matrix<1, 10> hidden2_b = gaussian_random_matrix<1, 10>();
  Matrix<10, 5> output_w = gaussian_random_matrix<10, 5>();
  Matrix<1, 5> output_b = gaussian_random_matrix<1, 5>();

  Matrix<NSAMPLE, 5> hidden1_z = x * hidden1_w + hidden1_b;
  Matrix<NSAMPLE, 5> hidden1_a = sigmoid(hidden1_z);
  Matrix<NSAMPLE, 10> hidden2_z = hidden1_a * hidden2_w + hidden2_b;
  Matrix<NSAMPLE, 10> hidden2_a = sigmoid(hidden2_z);
  Matrix<NSAMPLE, 5> output_z = hidden2_a * output_w + output_b;
  Matrix<NSAMPLE, 5> output_a = sigmoid(output_z);

  for (uint_fast32_t t = 0; t < 100000; ++t) {
    const Matrix<NSAMPLE, 5> output_delta = output_a - y;
    if (t % 100 == 0) {
      std::cout << t << ": "
                << hadamard_product(output_delta, output_delta).sum()
                << std::endl;
    }
    const Matrix<5, 10> hidden2_delta =
        hadamard_product(output_delta * output_w.transpose(), hidden2_a);
  }

  return 0;
}
