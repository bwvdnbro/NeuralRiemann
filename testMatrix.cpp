/*******************************************************************************
 * This file is part of NeuralRiemann
 * Copyright (C) 2018 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 *
 * NeuralRiemann is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * NeuralRiemann is distributed in the hope that it will be useful,
 * but WITOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with NeuralRiemann. If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

/**
 * @file testMatrix.cpp
 *
 * @brief Example showing how to use the Matrix class.
 *
 * To compile this program into an executable called `testMatrix`, run
 * ```
 *   g++ -O3 -std=c++11 -o testMatrix testMatrix.cpp
 * ```
 * (or similar for other compilers).
 *
 * @author Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 */
#include "Assert.hpp"
#include "Matrix.hpp"

/**
 * @brief Unit test for the Matrix class.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  Matrix<2, 2> a;
  const double barray[2][2] = {{1., 2.}, {3., 4.}};
  Matrix<2, 2> b(barray);
  const double carray[2][2] = {{1., 1.}, {2., 2.}};
  Matrix<2, 2> c(carray);

  a += b;

  assert_condition(a(0, 0) == 1.);
  assert_condition(a(0, 1) == 2.);
  assert_condition(a(1, 0) == 3.);
  assert_condition(a(1, 1) == 4.);

  a -= c;

  assert_condition(a(0, 0) == 0.);
  assert_condition(a(0, 1) == 1.);
  assert_condition(a(1, 0) == 1.);
  assert_condition(a(1, 1) == 2.);

  Matrix<2, 2> d = a * b;

  assert_condition(d(0, 0) == 3.);
  assert_condition(d(0, 1) == 4.);
  assert_condition(d(1, 0) == 7.);
  assert_condition(d(1, 1) == 10.);

  Matrix<2, 2> e = hadamard_product(a, b);

  assert_condition(e(0, 0) == 0.);
  assert_condition(e(0, 1) == 2.);
  assert_condition(e(1, 0) == 3.);
  assert_condition(e(1, 1) == 8.);

  e *= 2.;

  assert_condition(e(0, 0) == 0.);
  assert_condition(e(0, 1) == 4.);
  assert_condition(e(1, 0) == 6.);
  assert_condition(e(1, 1) == 16.);

  Matrix<2, 2> f = e * 0.5;

  assert_condition(f(0, 0) == 0.);
  assert_condition(f(0, 1) == 2.);
  assert_condition(f(1, 0) == 3.);
  assert_condition(f(1, 1) == 8.);

  Matrix<2, 2> g = 0.5 * e;

  assert_condition(g(0, 0) == 0.);
  assert_condition(g(0, 1) == 2.);
  assert_condition(g(1, 0) == 3.);
  assert_condition(g(1, 1) == 8.);

  assert_condition(g.sum() == 13.);

  return 0;
}
