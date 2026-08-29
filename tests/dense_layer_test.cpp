#include "nncpp/network.hpp"
#include "test_fixture.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace {
    void checkBackwardBoundary(const std::size_t inputSize) {
        constexpr std::size_t outputSize = 3;
        const nncpp::Tensor outputGradient = {0.5f, -0.75f, 1.25f};
        nncpp::DenseLayer layer(nncpp::Shape{inputSize}, nncpp::Shape{outputSize});
        nncpp::Tensor input(inputSize);
        nncpp::Tensor weights(inputSize * outputSize);

        for (std::size_t i = 0; i < inputSize; ++i) {
            input[i] = static_cast<nncpp::Scalar>(static_cast<int>(i % 9) - 4) * 0.125f;
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<nncpp::Scalar>(static_cast<int>(i % 13) - 6) * 0.0625f;
        }

        layer.setWeights(weights);
        layer.setBiases(nncpp::Tensor(outputSize, 0.0f));
        layer.resetGradients();

        static_cast<void>(layer.forward(input));
        const nncpp::Tensor firstInputGradient = layer.backward(outputGradient);
        static_cast<void>(layer.forward(input));
        const nncpp::Tensor secondInputGradient = layer.backward(outputGradient);

        for (std::size_t inIndex = 0; inIndex < inputSize; ++inIndex) {
            nncpp::Scalar expectedInputGradient = 0.0f;
            for (std::size_t outIndex = 0; outIndex < outputSize; ++outIndex) {
                const std::size_t weightIndex = outIndex * inputSize + inIndex;
                expectedInputGradient += weights[weightIndex] * outputGradient[outIndex];
                expectNear(
                    layer.getWeightGradients()[weightIndex],
                    2.0f * input[inIndex] * outputGradient[outIndex],
                    "accumulated boundary weight gradient"
                );
            }
            expectNear(firstInputGradient[inIndex], expectedInputGradient, "boundary input gradient");
            expectNear(secondInputGradient[inIndex], expectedInputGradient, "repeated boundary input gradient");
        }

        for (std::size_t outIndex = 0; outIndex < outputSize; ++outIndex) {
            expectNear(
                layer.getBiasGradients()[outIndex],
                2.0f * outputGradient[outIndex],
                "accumulated boundary bias gradient"
            );
        }
    }
}

int main() {
    constexpr std::array<std::size_t, 10> boundarySizes = {
        1, 7, 8, 9, 15, 16, 17, 31, 32, 33,
    };
    for (const std::size_t inputSize : boundarySizes) {
        checkBackwardBoundary(inputSize);
    }

    bool rejectedZeroDimension = false;
    try {
        static_cast<void>(nncpp::DenseLayer(nncpp::Shape{0}, nncpp::Shape{4}));
    } catch (const std::invalid_argument&) {
        rejectedZeroDimension = true;
    }
    if (!rejectedZeroDimension) {
        throw std::runtime_error("constructor accepted a zero dimension");
    }

    nncpp::DenseLayer layer(nncpp::Shape{2}, nncpp::Shape{4});
    if (layer.getInputShape() != nncpp::Shape{2} || layer.getOutputShape() != nncpp::Shape{4}) {
        throw std::runtime_error("dense layer shapes are incorrect");
    }
    constexpr nncpp::Scalar xavierLimit = 1.0f;
    for (const nncpp::Scalar weight : layer.getWeights()) {
        if (weight < -xavierLimit || weight > xavierLimit) {
            throw std::runtime_error("weight is outside Xavier range");
        }
    }
    for (const nncpp::Scalar gradient : layer.getWeightGradients()) {
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

    const nncpp::Tensor output = layer.forward({0.25f, 0.5f});
    constexpr std::array<double, 4> expectedOutput = {
        0.125, 0.125, -0.175, 0.575,
    };
    for (std::size_t index = 0; index < expectedOutput.size(); ++index) {
        expectNear(output[index], expectedOutput[index], "dense output");
    }

    const nncpp::Tensor inputGradient =
        layer.backward({1.0f, 2.0f, 3.0f, 4.0f});
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
        layer.applyGradient(0.0, 1);
    } catch (const std::invalid_argument&) {
        rejectedBadLearningRate = true;
    }
    if (!rejectedBadLearningRate) {
        throw std::runtime_error("applyGradient accepted zero learning rate");
    }

    const nncpp::Tensor weightsBefore = layer.getWeights();
    const nncpp::Tensor biasesBefore = layer.getBiases();
    constexpr nncpp::Scalar learningRate = 0.1f;
    layer.applyGradient(learningRate, 1);

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
    for (const nncpp::Scalar gradient : layer.getWeightGradients()) {
        expectNear(gradient, 0.0, "reset weight gradient");
    }
    for (const nncpp::Scalar gradient : layer.getBiasGradients()) {
        expectNear(gradient, 0.0, "reset bias gradient");
    }

    return 0;
}
