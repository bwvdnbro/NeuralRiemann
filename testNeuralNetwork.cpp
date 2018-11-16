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
 * @file testNeuralNetwork.cpp
 *
 * @brief Example showing how to use the neural network.
 *
 * To compile this program into an executable called `testNeuralNetwork`, run
 * ```
 *   g++ -O3 -std=c++11 -o testNeuralNetwork testNeuralNetwork.cpp
 * ```
 * (or similar for other compilers).
 *
 * @author Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 */
#include "NeuralNetwork.hpp"

/**
 * @brief Unit test for the NeuralNetwork class.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  NeuralNetwork<1, 20, 4> network;

  double x[1000], y[4000];
  for (uint_fast32_t i = 0; i < 1000; ++i) {
    double *xx = &x[i];
    xx[0] = random_double() * 100.;
    const uint_fast8_t neuron = 0.04 * xx[0];
    double *yy = &y[i * 4];
    yy[0] = 0.;
    yy[1] = 0.;
    yy[2] = 0.;
    yy[3] = 0.;
    yy[neuron] = 1.;
    const double result =
        std::round(yy[1]) + 2. * std::round(yy[2]) + 3. * std::round(yy[3]);
    std::cout << xx[0] << "\t" << result << " (" << yy[0] << ", " << yy[1]
              << ", " << yy[2] << ", " << yy[3] << ")\n";
  }

  network.train(x, y, 1000);

  for (uint_fast32_t i = 0; i < 100; ++i) {
    double xx[1], yy[4];
    xx[0] = i;
    network.evaluate(xx, yy);
    const double result =
        std::round(yy[1]) + 2. * std::round(yy[2]) + 3. * std::round(yy[3]);
    std::cout << i << "\t" << result << " (" << yy[0] << ", " << yy[1] << ", "
              << yy[2] << ", " << yy[3] << ")\n";
  }

  return 0;
}
