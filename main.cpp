#include "nncpp/network.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
    constexpr double learningRate = 0.01;
    constexpr double errorEpsilon = 1e-4;
    constexpr std::size_t maxTrainIterations = 1'000'000;

    nncpp::Network makeNetwork(const std::vector<double>& input) {
        nncpp::Network network(3);
        network[0].resize(2);
        network[1].resize(4);
        network[2].resize(1);

        if (input.size() != network.front().size()) {
            throw std::invalid_argument("input does not match input layer");
        }
        for (std::size_t index = 0; index < input.size(); ++index) {
            network.front()[index].activation = input[index];
        }

        network[1][0].weights = {0.1, 0.2};
        network[1][1].weights = {-0.3, 0.4};
        network[1][1].bias = 0.1;
        network[1][2].weights = {0.5, -0.6};
        network[1][2].bias = -0.1;
        network[1][3].weights = {0.7, 0.8};
        network[1][3].bias = 0.2;

        network[2][0].weights = {0.2, -0.1, 0.4, 0.3};
        network[2][0].bias = -0.2;

        return network;
    }
}

static void simpleNetworkExample() {
    const std::vector input = {0.25, 0.5};
    const std::vector targets = {1.0};

    nncpp::Network network = makeNetwork(input);

    std::size_t iterations = 0;
    for (; iterations < maxTrainIterations; ++iterations) {
        nncpp::forwardPass(network);
        if (nncpp::loss(network.back(), targets) < errorEpsilon) {
            break;
        }
        nncpp::calculateDeltas(network, targets);
        nncpp::optimizeParams(network, learningRate);
    }

    nncpp::forwardPass(network);
    const double finalError = nncpp::loss(network.back(), targets);

    std::cout << "iterations: " << iterations << '\n';
    std::cout << "output: " << network.back().front().activation << '\n';
    std::cout << "error: " << finalError << '\n';

    if (finalError >= errorEpsilon) {
        throw std::runtime_error("training did not converge");
    }
}

static void newNetworkExample() {
    const std::vector input = {0.25, 0.5};
    const std::vector targets = {1.0};

    nncpp::DenseLayer dense1(2, 4);
    nncpp::Sigmoid sigmoid1(4);
    nncpp::DenseLayer dense2(4, 1);
    nncpp::Sigmoid sigmoid2(1);

    size_t iterations = 0;
    for (; iterations < maxTrainIterations; ++iterations) {
        const auto z1 = dense1.forward(input);
        const auto a1 = sigmoid1.forward(z1);
        const auto z2 = dense2.forward(a1);
        const auto output = sigmoid2.forward(z2);

        if (nncpp::loss(output, targets) < errorEpsilon) {
            break;
        }

        dense1.resetGradients();
        dense2.resetGradients();

        auto gradient = nncpp::derivativeMSE(output, targets);
        gradient = sigmoid2.backward(gradient);
        gradient = dense2.backward(gradient);
        gradient = sigmoid1.backward(gradient);
        const auto inputGradient = dense1.backward(gradient);
        static_cast<void>(inputGradient);

        dense2.applyGradient(learningRate);
        dense1.applyGradient(learningRate);
    }
    const auto z1 = dense1.forward(input);
    const auto a1 = sigmoid1.forward(z1);
    const auto z2 = dense2.forward(a1);
    const auto output = sigmoid2.forward(z2);

    const double finalError = nncpp::loss(output, targets);
    std::cout << "iterations: " << iterations << '\n';
    std::cout << "output: " << output[0] << '\n';
    std::cout << "error: " << finalError << '\n';

    if (finalError >= errorEpsilon) {
        throw std::runtime_error("training did not converge");
    }
}

int main() {
    simpleNetworkExample();
    newNetworkExample();
    return 0;
}
