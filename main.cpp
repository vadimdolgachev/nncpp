#include "nncpp/network.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace {
    constexpr nncpp::Scalar learningRate = 0.01f;
    constexpr nncpp::Scalar errorEpsilon = 1e-4f;
    constexpr std::size_t maxTrainIterations = 1'000'000;

    std::vector<std::vector<nncpp::Neuron> > makeNetwork(const nncpp::Tensor &input) {
        std::vector<std::vector<nncpp::Neuron> > network(3);
        network[0].resize(2);
        network[1].resize(4);
        network[2].resize(1);

        if (input.size() != network.front().size()) {
            throw std::invalid_argument("input does not match input layer");
        }
        for (std::size_t index = 0; index < input.size(); ++index) {
            network.front()[index].activation = input[index];
        }

        network[1][0].weights = {0.1f, 0.2f};
        network[1][1].weights = {-0.3f, 0.4f};
        network[1][1].bias = 0.1f;
        network[1][2].weights = {0.5f, -0.6f};
        network[1][2].bias = -0.1f;
        network[1][3].weights = {0.7f, 0.8f};
        network[1][3].bias = 0.2f;

        network[2][0].weights = {0.2f, -0.1f, 0.4f, 0.3f};
        network[2][0].bias = -0.2f;

        return network;
    }

    void simpleNetworkExample() {
        const nncpp::Tensor input = {0.25, 0.5};
        const nncpp::Tensor targets = {1.0};

        std::vector<std::vector<nncpp::Neuron> > network = makeNetwork(input);

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
        const nncpp::Scalar finalError = nncpp::loss(network.back(), targets);

        std::cout << "iterations: " << iterations << '\n';
        std::cout << "output: " << network.back().front().activation << '\n';
        std::cout << "error: " << finalError << '\n';

        if (finalError >= errorEpsilon) {
            throw std::runtime_error("training did not converge");
        }
    }

    void newNetworkExample() {
        const nncpp::Tensor input = {0.25, 0.5};
        const nncpp::Tensor targets = {1.0};

        std::array<std::unique_ptr<nncpp::Layer>, 4> network = {
           std::make_unique<nncpp::DenseLayer>(2, 4),
            std::make_unique<nncpp::Sigmoid>(4),
            std::make_unique<nncpp::DenseLayer>(4, 1),
            std::make_unique<nncpp::Sigmoid>(1),
        };

        size_t iterations = 0;
        for (; iterations < maxTrainIterations; ++iterations) {
            nncpp::Tensor out = input;
            for (const auto &layer: network) {
                out = layer->forward(out);
            }

            if (nncpp::loss(out, targets) < errorEpsilon) {
                break;
            }

            for (const auto &layer: network) {
                layer->resetGradients();
            }

            auto gradient = nncpp::derivativeMSE(out, targets);

            for (const auto &it : std::views::reverse(network)) {
                gradient = it->backward(gradient);
            }

            for (const auto &layer: network) {
                layer->applyGradient(learningRate);
            }
        }

        nncpp::Tensor out = input;
        for (const auto &layer: network) {
            out = layer->forward(out);
        }
        const nncpp::Scalar finalError = nncpp::loss(out, targets);
        std::cout << "iterations: " << iterations << '\n';
        std::cout << "output: " << out[0] << '\n';
        std::cout << "error: " << finalError << '\n';

        if (finalError >= errorEpsilon) {
            throw std::runtime_error("training did not converge");
        }
    }
}

int main() {
    simpleNetworkExample();
    newNetworkExample();
    return 0;
}
