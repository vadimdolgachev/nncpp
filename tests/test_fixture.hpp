#pragma once

#include "nncpp/network.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

inline std::vector<std::vector<nncpp::Neuron>> makeTestNetwork() {
    std::vector<std::vector<nncpp::Neuron>> network(3);
    network[0].resize(2);
    network[1].resize(4);
    network[2].resize(1);

    network[0][0].activation = 0.25;
    network[0][1].activation = 0.5;

    network[1][0].weights = {0.1, 0.2};
    network[1][1].weights = {-0.3, 0.4};
    network[1][1].bias = 0.1f;
    network[1][2].weights = {0.5, -0.6};
    network[1][2].bias = -0.1f;
    network[1][3].weights = {0.7, 0.8};
    network[1][3].bias = 0.2f;

    network[2][0].weights = {0.2, -0.1, 0.4, 0.3};
    network[2][0].bias = -0.2f;

    return network;
}

inline void expectNear(const nncpp::Scalar actual,
                       const double expected,
                       const std::string &label,
                       const double tolerance = 1e-6) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(label + " does not match expected value");
    }
}
