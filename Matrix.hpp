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
 * @file Matrix.hpp
 *
 * @brief General matrix representation.
 *
 * @author Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 */
#ifndef MATRIX_HPP
#define MATRIX_HPP

/**
 * @brief _M_ x _N_ matrix.
 */
template <uint_fast32_t _M_, uint_fast32_t _N_> class Matrix {
private:
  double _matrix[_M_][_N_];

public:
  /**
   * @brief Empty constructor.
   */
  inline Matrix() {
    for (uint_fast32_t m = 0; m < _M_; ++m) {
      for (uint_fast32_t n = 0; n < _N_; ++n) {
        _matrix[m][n] = 0.;
      }
    }
  }

  /**
   * @brief Array constructor.
   *
   * @param array Array representation of the matrix.
   */
  inline Matrix(const double array[_M_][_N_]) {
    for (uint_fast32_t m = 0; m < _M_; ++m) {
      for (uint_fast32_t n = 0; n < _N_; ++n) {
        _matrix[m][n] = array[m][n];
      }
    }
  }

  /**
   * @brief Get the contents of the cell on row m and column n.
   *
   * @param m Row index.
   * @param n Column index.
   * @return Corresponding matrix element.
   */
  inline double operator()(const uint_fast32_t m, const uint_fast32_t n) const {
    return _matrix[m][n];
  }

  /**
   * @brief Access the contents of the cell on row m and column n.
   *
   * @param m Row index.
   * @param n Column index.
   * @return Reference to the corresponding matrix element.
   */
  inline double &operator()(const uint_fast32_t m, const uint_fast32_t n) {
    return _matrix[m][n];
  }

  /**
   * @brief Add the given matrix to this matrix.
   *
   * @param matrix Matrix to add.
   * @return Reference to the incremented matrix.
   */
  inline Matrix &operator+=(const Matrix<_M_, _N_> &matrix) {
    for (uint_fast32_t m = 0; m < _M_; ++m) {
      for (uint_fast32_t n = 0; n < _N_; ++n) {
        _matrix[m][n] += matrix._matrix[m][n];
      }
    }
    return *this;
  }

  /**
   * @brief Subtract the given matrix from this matrix.
   *
   * @param matrix Matrix to subtract.
   * @return Reference to the decremented matrix.
   */
  inline Matrix &operator-=(const Matrix<_M_, _N_> &matrix) {
    for (uint_fast32_t m = 0; m < _M_; ++m) {
      for (uint_fast32_t n = 0; n < _N_; ++n) {
        _matrix[m][n] -= matrix._matrix[m][n];
      }
    }
    return *this;
  }
};

template <uint_fast32_t _M_, uint_fast32_t _N_, uint_fast32_t _O_>
static inline Matrix<_M_, _O_> operator*(const Matrix<_M_, _N_> &a,
                                         const Matrix<_N_, _O_> &b) {
  Matrix<_M_, _O_> c;
  for (uint_fast32_t m = 0; m < _M_; ++m) {
    for (uint_fast32_t o = 0; o < _O_; ++o) {
      for (uint_fast32_t n = 0; n < _N_; ++n) {
        c(m, o) += a(m, n) * b(n, o);
      }
    }
  }
  return c;
}

#endif // MATRIX_HPP
