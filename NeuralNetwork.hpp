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
 * @file NeuralNetwork.hpp
 *
 * @brief Neural network.
 *
 * @author Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 */
#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <cinttypes>
#include <cmath>
#include <fstream>
#include <iostream>

/*! @brief Number of training epochs. */
#define TRAINING_SIZE 1e5

/*! @brief Decay factor for neuron weights (L2 regularization). */
#define WEIGHT_DECAY 1.

/*! @brief Network learning rate. */
#define LEARNING_RATE 1.0

/**
 * @brief Generate a uniformly distributed pseudo random number.
 *
 * @return Uniformly distributed pseudo random number in the range [0, 1].
 */
inline double random_double() { return ((double)rand()) / ((double)RAND_MAX); }

/**
 * @brief Generate _size_ normal distributed pseudo random numbers with an
 * average of 0 and standard deviation of 1.
 *
 * We use the Box-Muller method to generate pairs of pseudo random numbers and
 * throw away the last one if _size_ is odd.
 *
 * @param x Array to store the results in.
 */
template <uint_fast32_t _size_>
inline void gaussian_random_numbers(double x[_size_]) {
  for (uint_fast32_t i = 0; 2 * i < _size_; ++i) {
    const double u1 = std::sqrt(-2. * std::log(random_double()));
    const double u2 = 2. * M_PI * random_double();
    x[2 * i] = u1 * std::cos(u2);
    if (2 * i + 1 < _size_) {
      x[2 * i + 1] = u1 * std::sin(u2);
    }
  }
}

/**
 * @brief Sigmoid function.
 *
 * @param x Input value.
 * @return Output value.
 */
inline double sigmoid(const double x) { return 1. / (1. + std::exp(-x)); }

/**
 * @brief Neuron representation with _input_size_ connections in the previous
 * layer.
 */
template <uint_fast32_t _input_size_> class Neuron {
private:
  /// Network variables
  /*! @brief Input weights. */
  double _weight[_input_size_];
  /*! @brief Neuron bias. */
  double _bias;

  /// Training variables
  /*! @brief Input activations. */
  double _input_activation[_input_size_];
  /*! @brief Output activation. */
  double _output_activation;
  /*! @brief Gradient factors for the weights. */
  double _delta_weight[_input_size_];
  /*! @brief Gradient factor for the bias. */
  double _delta_bias;

public:
  /**
   * @brief Constructor.
   */
  inline Neuron() {
    double u[_input_size_ + 1];
    gaussian_random_numbers<_input_size_ + 1>(u);
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      _weight[i] = u[i];
    }
    _bias = u[_input_size_ - 1];
  }

  /// Network functionality

  /**
   * @brief Evaluate the neuron contribution using the given input from the
   * previous layer.
   *
   * @param input Input from the previous layer.
   * @return Output from this neuron.
   */
  inline double evaluate(const double input[_input_size_]) const {
    double result = _bias;
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      result += _weight[i] * input[i];
    }
    return sigmoid(result);
  }

  /// Training functionality

  /**
   * @brief Evaluate the neuron contribution and store variables for training.
   *
   * @param input Input from the previous layer.
   * @return Output from this neuron.
   */
  inline double training_evaluate(const double input[_input_size_]) {
    double result = _bias;
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      _input_activation[i] = input[i];
      result += _weight[i] * input[i];
    }
    _output_activation = sigmoid(result);
    return _output_activation;
  }

  /**
   * @brief Backpropagate the network error using the given input from the next
   * layer.
   *
   * @param delta_in Error input from the next layer.
   * @param delta_out Error output to the previous layer (is appended).
   */
  inline void delta_propagate(const double delta_in,
                              double delta_out[_input_size_]) {
    const double dsigma = _output_activation * (1. - _output_activation);
    const double delta = delta_in * dsigma;
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      delta_out[i] += delta * _weight[i];
      _delta_weight[i] += delta * _input_activation[i];
    }
    _delta_bias += delta;
  }

  /**
   * @brief Initialize the training variables for this neuron.
   */
  inline void initialize_training() {
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      _delta_weight[i] = 0.;
    }
    _delta_bias = 0.;
  }

  /**
   * @brief Update the neuron weights and bias after a training step completed.
   *
   * @param eta Normalized learning rate.
   */
  inline void update(const double eta) {
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      _weight[i] *= (1. - eta * WEIGHT_DECAY);
      _weight[i] -= eta * _delta_weight[i];
    }
    _bias -= eta * _delta_bias;
  }

  /**
   * @brief Get the weights and bias for output purposes.
   *
   * @param weight Array to store the weights in.
   * @param bias Variable to store the bias in.
   */
  inline void get_contents(double weight[_input_size_], double *bias) const {
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      weight[i] = _weight[i];
    }
    *bias = _bias;
  }
};

/**
 * @brief Single layer of the neural network, of size _output_size_ and with
 * _input_size_ the size of the previous layer.
 */
template <uint_fast32_t _input_size_, uint_fast32_t _output_size_> class Layer {
private:
  /*! @brief Neurons in this layer. */
  Neuron<_input_size_> _neurons[_output_size_];

public:
  /// Network functionality

  /**
   * @brief Evaluate the contribution of this layer, based on the given input
   * from the previous layer.
   *
   * @param input Input from the previous layer.
   * @param output Contribution of this layer.
   */
  inline void evaluate(const double input[_input_size_],
                       double output[_output_size_]) const {
    for (uint_fast32_t i = 0; i < _output_size_; ++i) {
      output[i] = _neurons[i].evaluate(input);
    }
  }

  /// Training functionality

  /**
   * @brief Evaluate the contribution from this layer and store variables for
   * training.
   *
   * @param input Input from the previous layer.
   * @param output Output from this layer.
   */
  inline void training_evaluate(const double input[_input_size_],
                                double output[_output_size_]) {
    for (uint_fast32_t i = 0; i < _output_size_; ++i) {
      output[i] = _neurons[i].training_evaluate(input);
    }
  }

  /**
   * @brief Backpropagate the error of the network using the input from the next
   * layer.
   *
   * @param delta_in Error input from the next layer.
   * @param delta_out Error output to the previous layer.
   */
  inline void delta_propagate(const double delta_in[_output_size_],
                              double delta_out[_input_size_]) {
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      delta_out[i] = 0.;
    }
    for (uint_fast32_t i = 0; i < _output_size_; ++i) {
      _neurons[i].delta_propagate(delta_in[i], delta_out);
    }
  }

  /**
   * @brief Initialize the training variables for this layer.
   */
  inline void initialize_training() {
    for (uint_fast32_t i = 0; i < _output_size_; ++i) {
      _neurons[i].initialize_training();
    }
  }

  /**
   * @brief Update the neurons in this layer after a training step.
   *
   * @param eta Normalized learning rate.
   */
  inline void update(const double eta) {
    for (uint_fast32_t i = 0; i < _output_size_; ++i) {
      _neurons[i].update(eta);
    }
  }

  /**
   * @brief Add the contents of this layer to the given output stream in dot
   * format.
   *
   * @param steam Stream to write to.
   * @param input_connections Node ids of the nodes in the previous layer.
   * @param output_connections Array to store the node ids in this layer in.
   * @param nodecount Node id counter (is updated).
   */
  inline void get_contents(std::ostream &stream,
                           const uint_fast32_t input_connections[_input_size_],
                           uint_fast32_t output_connections[_output_size_],
                           uint_fast32_t &nodecount) const {

    for (uint_fast32_t i = 0; i < _output_size_; ++i) {
      double weight[_input_size_], bias;
      _neurons[i].get_contents(weight, &bias);
      stream << "n" << nodecount << " [label=\"" << bias << "\"];\n";
      for (uint_fast32_t j = 0; j < _input_size_; ++j) {
        stream << "n" << input_connections[j] << " -> n" << nodecount
               << " [label=\"" << weight[j] << "\"];\n";
      }
      output_connections[i] = nodecount;
      ++nodecount;
    }
  }
};

/**
 * @brief Neural network with _hidden_size_ neurons in 1 hidden layer.
 */
template <uint_fast32_t _input_size_, uint_fast32_t _hidden_size_,
          uint_fast32_t _output_size_>
class NeuralNetwork {
private:
  /*! @brief First layer. */
  Layer<_input_size_, _hidden_size_> _layer0;
  /*! @brief Second layer. */
  Layer<_hidden_size_, _output_size_> _layer1;

public:
  /// Network functionality

  /**
   * @brief Evaluate the network.
   *
   * @param input Input value(s).
   * @param output Output value(s).
   */
  inline void evaluate(const double input[_input_size_],
                       double output[_output_size_]) const {
    double layer0_out[_hidden_size_], layer1_out[_output_size_];
    _layer0.evaluate(input, layer0_out);
    _layer1.evaluate(layer0_out, layer1_out);

    output[0] = layer1_out[0];
  }

  /// Training functionality

  /**
   * @brief Train the network on the given data.
   *
   * @param x Input training values.
   * @param y Expected output training values.
   * @return size Size of the training data.
   */
  inline void train(const double *x, const double *y,
                    const uint_fast32_t size) {

    for (uint_fast32_t epoch = 0; epoch < TRAINING_SIZE; ++epoch) {
      // initialize the deltas for all layers
      _layer0.initialize_training();
      _layer1.initialize_training();

      double error = 0.;
      // loop over all training data
      for (uint_fast32_t ival = 0; ival < size; ++ival) {
        double layer0_out[_hidden_size_], layer1_out[_output_size_];
        _layer0.training_evaluate(&x[ival * _input_size_], layer0_out);
        _layer1.training_evaluate(layer0_out, layer1_out);
        double delta_ent[_output_size_];
        for (uint_fast32_t i = 0; i < _output_size_; ++i) {
          double yout = layer1_out[i];
          const double yin = y[ival * _output_size_ + i];
          const double ydiff = yout - y[ival * _output_size_ + i];
          error += 0.5 * ydiff * ydiff;
          delta_ent[i] = ydiff;
        }
        double layer1_delta[_hidden_size_], layer0_delta[_input_size_];
        _layer1.delta_propagate(delta_ent, layer1_delta);
        _layer0.delta_propagate(layer1_delta, layer0_delta);
      }

      if (epoch % 100 == 0) {
        std::cout << epoch << "\t" << error << " ("
                  << (epoch * 100. / TRAINING_SIZE) << "%)\n";
      }

      const double eta = LEARNING_RATE / size;
      // update the weights and biases
      _layer0.update(eta);
      _layer1.update(eta);
    }
  }

  /**
   * @brief Write the neural network to a file with the given name, in dot
   * format.
   *
   * @param filename Name of the file to write.
   */
  inline void print_dot(const std::string filename) const {

    uint_fast32_t nodecount = 0;
    std::ofstream ofile(filename);
    uint_fast32_t layer0_input[_input_size_], layer0_output[_hidden_size_],
        layer1_output[_output_size_];
    ofile << "digraph {\n";
    for (uint_fast32_t i = 0; i < _input_size_; ++i) {
      ofile << "n" << nodecount << "[label=\"input\"];\n";
      layer0_input[i] = nodecount;
      ++nodecount;
    }
    _layer0.get_contents(ofile, layer0_input, layer0_output, nodecount);
    _layer1.get_contents(ofile, layer0_output, layer1_output, nodecount);
    ofile << "}";
  }
};

#endif // NEURALNETWORK_HPP
