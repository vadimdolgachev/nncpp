#include "nncpp/network.hpp"
#include "test_fixture.hpp"

#include <array>
#include <stdexcept>
#include <vector>

int main() {
    bool rejectedZeroDimension = false;
    try {
        static_cast<void>(nncpp::DenseLayer(0, 4));
    } catch (const std::invalid_argument&) {
        rejectedZeroDimension = true;
    }
    if (!rejectedZeroDimension) {
        throw std::runtime_error("constructor accepted a zero dimension");
    }

    nncpp::DenseLayer layer(2, 4);
    constexpr double xavierLimit = 1.0;
    for (const double weight : layer.getWeights()) {
        if (weight < -xavierLimit || weight > xavierLimit) {
            throw std::runtime_error("weight is outside Xavier range");
        }
    }
    for (const double gradient : layer.getWeightGradients()) {
        expectNear(gradient, 0.0, "initial weight gradient");
    }
    for (std::size_t index = 0; index < layer.getBiases().size(); ++index) {
        expectNear(layer.getBiases()[index], 0.0, "initial bias");
        expectNear(
            layer.getBiasGradients()[index],
            0.0,
            "initial bias gradient"
        );
    }

    bool rejectedBadWeights = false;
    try {
        layer.setWeights({1.0});
    } catch (const std::invalid_argument&) {
        rejectedBadWeights = true;
    }
    if (!rejectedBadWeights) {
        throw std::runtime_error("setWeights accepted an invalid size");
    }

    bool rejectedBadBiases = false;
    try {
        layer.setBiases({1.0});
    } catch (const std::invalid_argument&) {
        rejectedBadBiases = true;
    }
    if (!rejectedBadBiases) {
        throw std::runtime_error("setBiases accepted an invalid size");
    }

    layer.setWeights({
        0.1, 0.2,
        -0.3, 0.4,
        0.5, -0.6,
        0.7, 0.8,
    });

    bool rejectedMissingForward = false;
    try {
        static_cast<void>(layer.backward({1.0, 2.0, 3.0, 4.0}));
    } catch (const std::logic_error&) {
        rejectedMissingForward = true;
    }
    if (!rejectedMissingForward) {
        throw std::runtime_error("backward accepted a missing forward pass");
    }

    const std::vector<double> output = layer.forward({0.25, 0.5});
    constexpr std::array<double, 4> expectedOutput = {
        0.125, 0.125, -0.175, 0.575,
    };
    for (std::size_t index = 0; index < expectedOutput.size(); ++index) {
        expectNear(output[index], expectedOutput[index], "dense output");
    }

    const std::vector<double> inputGradient =
        layer.backward({1.0, 2.0, 3.0, 4.0});
    expectNear(inputGradient[0], 3.8, "input gradient 0");
    expectNear(inputGradient[1], 2.4, "input gradient 1");

    constexpr std::array<double, 8> expectedWeightGradients = {
        0.25, 0.5,
        0.5, 1.0,
        0.75, 1.5,
        1.0, 2.0,
    };
    for (std::size_t index = 0;
         index < expectedWeightGradients.size();
         ++index) {
        expectNear(
            layer.getWeightGradients()[index],
            expectedWeightGradients[index],
            "weight gradient"
        );
    }

    constexpr std::array<double, 4> expectedBiasGradients = {
        1.0, 2.0, 3.0, 4.0,
    };
    for (std::size_t index = 0;
         index < expectedBiasGradients.size();
         ++index) {
        expectNear(
            layer.getBiasGradients()[index],
            expectedBiasGradients[index],
            "bias gradient"
        );
    }

    bool rejectedBadLearningRate = false;
    try {
        layer.applyGradient(0.0);
    } catch (const std::invalid_argument&) {
        rejectedBadLearningRate = true;
    }
    if (!rejectedBadLearningRate) {
        throw std::runtime_error("applyGradient accepted zero learning rate");
    }

    const std::vector<double> weightsBefore = layer.getWeights();
    const std::vector<double> biasesBefore = layer.getBiases();
    constexpr double learningRate = 0.1;
    layer.applyGradient(learningRate);

    for (std::size_t index = 0; index < weightsBefore.size(); ++index) {
        expectNear(
            layer.getWeights()[index],
            weightsBefore[index] -
                learningRate * expectedWeightGradients[index],
            "updated dense weight"
        );
        expectNear(
            layer.getWeightGradients()[index],
            expectedWeightGradients[index],
            "preserved weight gradient"
        );
    }
    for (std::size_t index = 0; index < biasesBefore.size(); ++index) {
        expectNear(
            layer.getBiases()[index],
            biasesBefore[index] -
                learningRate * expectedBiasGradients[index],
            "updated dense bias"
        );
    }

    layer.resetGradients();
    for (const double gradient : layer.getWeightGradients()) {
        expectNear(gradient, 0.0, "reset weight gradient");
    }
    for (const double gradient : layer.getBiasGradients()) {
        expectNear(gradient, 0.0, "reset bias gradient");
    }

    return 0;
}
