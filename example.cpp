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
 * @file example.cpp
 *
 * @brief Example showing how to use the exact Riemann solver.
 *
 * To compile this program into an executable called `example`, run
 * ```
 *   g++ -O3 -std=c++11 -o example example.cpp
 * ```
 * (or similar for other compilers).
 *
 * @author Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 */
#include "Assert.hpp"
#include "NeuralNetwork.hpp"
#include "RiemannSolver.hpp"

#include <fstream>

/**
 * @brief Run a Rieman solver test.
 *
 * @param solver RiemannSolver to test.
 * @param rhoL Density of the left state.
 * @param uL Velocity of the left state.
 * @param PL Pressure of the left state.
 * @param rhoR Density of the right state.
 * @param uR Velocity of the right state.
 * @param PR Pressure of the right state.
 * @param rhoexp Expected density solution.
 * @param uexp Expected velocity solution.
 * @param Pexp Expected pressure solution.
 * @param expstruct Expected solution structure.
 */
void run_test(RiemannSolver &solver, double rhoL, double uL, double PL,
              double rhoR, double uR, double PR, double rhoexp, double uexp,
              double Pexp, RiemannSolver::RiemannSolutionStructure expstruct) {

  nnr_status("Running Riemann solver test with input values WL = (%g %g %g) "
             "and WR = (%g %g %g)...",
             rhoL, uL, PL, rhoR, uR, PR);

  double rhosol, usol, Psol;
  RiemannSolver::RiemannSolutionStructure solstruct =
      solver.solve(rhoL, uL, PL, rhoR, uR, PR, rhosol, usol, Psol);
  assert_values_equal_rel(rhosol, rhoexp, 1.e-4);
  assert_values_equal_rel(usol, uexp, 1.e-4);
  assert_values_equal_rel(Psol, Pexp, 1.e-4);
  assert_condition(solstruct == expstruct);

  // symmetry test: also test the problem with left and right states reversed
  solver.solve(rhoR, -uR, PR, rhoL, -uL, PL, rhosol, usol, Psol);
  assert_values_equal_rel(rhosol, rhoexp, 1.e-4);
  assert_values_equal_rel(usol, -uexp, 1.e-4);
  assert_values_equal_rel(Psol, Pexp, 1.e-4);

  nnr_status("Got expected result.");
}

/**
 * @brief Unit test for the RiemannSolver class.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  nnr_status("Running Riemann solver tests...");

  /// standard tests
  {
    RiemannSolver solver(5. / 3.);

    // Toro tests
    run_test(solver, 1., 0., 1., 0.125, 0., 0.1, 0.47969, 0.841194, 0.293945,
             RiemannSolver::RAREFACTION_SHOCK);
    run_test(solver, 1., -2., 0.4, 1., 2., 0.4, 0.00617903, 0., 8.32249e-05,
             RiemannSolver::RAREFACTION_RAREFACTION);
    run_test(solver, 1., 0., 1000., 1., 0., 0.01, 0.615719, 18.2812, 445.626,
             RiemannSolver::RAREFACTION_SHOCK);
    run_test(solver, 1., 0., 0.01, 1., 0., 100., 0.61577, -5.78011, 44.5687,
             RiemannSolver::SHOCK_RAREFACTION);
    run_test(solver, 5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950,
             12.743, 8.56045, 1841.82, RiemannSolver::SHOCK_SHOCK);
    // vacuum generation
    run_test(solver, 1., -1., 1.e-6, 1., 1., 1.0005e-6, 0., 0., 0.,
             RiemannSolver::VACUUM);
  }

  nnr_status("All tests successfully finished!");

  /// generate training data for neural network
  RiemannSolver solver(5. / 3.);
  std::ofstream ofile("training_data.txt");
  ofile << "# rhoL\tuL\tPL\trhoR\tuR\tPR\tstructure\n";
  uint_fast32_t ndone[5] = {0, 0, 0, 0, 0};
  for (uint_fast32_t i = 0; i < 10000; ++i) {
    double rhoL = random_double() * 100.;
    double rhoR = random_double() * 100.;
    double uL = (random_double() - 0.5) * 200.;
    double uR = (random_double() - 0.5) * 200.;
    double PL = random_double() * 100.;
    double PR = random_double() * 100.;

    double rhosol, usol, Psol;
    RiemannSolver::RiemannSolutionStructure solstruct =
        solver.solve(rhoL, uL, PL, rhoR, uR, PR, rhosol, usol, Psol);
    while (ndone[solstruct] == 2000) {
      rhoL = random_double() * 100.;
      rhoR = random_double() * 100.;
      uL = (random_double() - 0.5) * 200.;
      uR = (random_double() - 0.5) * 200.;
      PL = random_double() * 100.;
      PR = random_double() * 100.;

      solstruct = solver.solve(rhoL, uL, PL, rhoR, uR, PR, rhosol, usol, Psol);
    }

    ++ndone[solstruct];

    ofile << rhoL << "\t" << uL << "\t" << PL << "\t" << rhoR << "\t" << uR
          << "\t" << PR << "\t" << solstruct << "\n";
  }

  return 0;
}
