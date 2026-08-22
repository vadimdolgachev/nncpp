#include "nncpp/network.hpp"
#include "test_fixture.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
    template<typename Exception, typename Callable>
    void expectThrows(Callable action, const std::string &label) {
        try {
            action();
        } catch (const Exception &) {
            return;
        }
        throw std::runtime_error(label + " did not throw");
    }

    void expectTensor(const nncpp::Tensor &actual,
                      const nncpp::Tensor &expected,
                      const std::string &label) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error(label + " has an unexpected size");
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            expectNear(actual[i], expected[i], label);
        }
    }
}

int main() {
    using Padding = nncpp::Conv2d::Padding;

    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(0, 1, 3, 3, 1, 1, Padding::Valid)); },
        "zero input channels"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(1, 1, 3, 3, 4, 1, Padding::Valid)); },
        "kernel larger than valid input"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(1, 1, 3, 3, 2, 1, Padding::Same)); },
        "even same-padding kernel"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(1, 1, 3, 3, 3, 2, Padding::Same)); },
        "strided same padding"
    );

    nncpp::Conv2d layer(2, 1, 3, 2, 2, 1, Padding::Valid);
    if (layer.getOutputWidth() != 2 || layer.getOutputHeight() != 1 || layer.getOutputSize() != 2) {
        throw std::runtime_error("Conv2d output shape is incorrect before forward");
    }

    const nncpp::Scalar xavierLimit = std::sqrt(6.0f / 12.0f);
    for (const nncpp::Scalar weight : layer.getWeights()) {
        if (weight < -xavierLimit || weight > xavierLimit) {
            throw std::runtime_error("Conv2d weight is outside Xavier range");
        }
    }
    expectTensor(layer.getWeightGradients(), nncpp::Tensor(8, 0.0f), "initial weight gradients");
    expectTensor(layer.getBiases(), {0.0f}, "initial biases");
    expectTensor(layer.getBiasGradients(), {0.0f}, "initial bias gradients");

    expectThrows<std::invalid_argument>(
        [&layer] { layer.setWeights({1.0f}); },
        "mismatched weights"
    );
    expectThrows<std::invalid_argument>(
        [&layer] { layer.setBiases({1.0f, 2.0f}); },
        "mismatched biases"
    );
    expectThrows<std::logic_error>(
        [&layer] { static_cast<void>(layer.forward({1.0f})); },
        "mismatched input"
    );
    expectThrows<std::logic_error>(
        [&layer] { static_cast<void>(layer.backward({1.0f, 1.0f})); },
        "backward before forward"
    );

    const nncpp::Tensor initialWeights = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        10.0f, 0.0f,
        0.0f, 10.0f,
    };
    layer.setWeights(initialWeights);
    layer.setBiases({0.5f});

    const nncpp::Tensor input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
    };
    expectTensor(layer.forward(input), {186.5f, 208.5f}, "rectangular multi-channel output");
    expectThrows<std::logic_error>(
        [&layer] { static_cast<void>(layer.backward({1.0f})); },
        "mismatched output gradient"
    );

    const nncpp::Tensor inputGradient = layer.backward({2.0f, 3.0f});
    expectTensor(
        inputGradient,
        {
            2.0f, 3.0f, 0.0f,
            0.0f, 2.0f, 3.0f,
            20.0f, 30.0f, 0.0f,
            0.0f, 20.0f, 30.0f,
        },
        "input gradient"
    );

    const nncpp::Tensor expectedWeightGradients = {
        8.0f, 13.0f,
        23.0f, 28.0f,
        38.0f, 43.0f,
        53.0f, 58.0f,
    };
    expectTensor(layer.getWeightGradients(), expectedWeightGradients, "weight gradients");
    expectTensor(layer.getBiasGradients(), {5.0f}, "bias gradients");
    expectThrows<std::logic_error>(
        [&layer] { static_cast<void>(layer.backward({2.0f, 3.0f})); },
        "repeated backward"
    );

    expectThrows<std::invalid_argument>(
        [&layer] { layer.applyGradient(0.0f, 2); },
        "zero learning rate"
    );
    expectThrows<std::invalid_argument>(
        [&layer] { layer.applyGradient(0.1f, 0); },
        "zero batch size"
    );

    constexpr nncpp::Scalar learningRate = 0.1f;
    constexpr std::size_t batchSize = 2;
    layer.applyGradient(learningRate, batchSize);
    const nncpp::Scalar updateScale = learningRate / static_cast<nncpp::Scalar>(batchSize);
    for (std::size_t i = 0; i < initialWeights.size(); ++i) {
        expectNear(
            layer.getWeights()[i],
            initialWeights[i] - updateScale * expectedWeightGradients[i],
            "updated convolution weight"
        );
    }
    expectNear(layer.getBiases()[0], 0.25, "updated convolution bias");

    layer.resetGradients();
    expectTensor(layer.getWeightGradients(), nncpp::Tensor(8, 0.0f), "reset weight gradients");
    expectTensor(layer.getBiasGradients(), {0.0f}, "reset bias gradients");

    nncpp::Conv2d samePadding(1, 1, 2, 2, 5, 1, Padding::Same);
    nncpp::Tensor centerKernel(25, 0.0f);
    centerKernel[12] = 1.0f;
    samePadding.setWeights(centerKernel);
    samePadding.setBiases({0.0f});
    if (samePadding.getOutputWidth() != 2 || samePadding.getOutputHeight() != 2) {
        throw std::runtime_error("same padding did not preserve spatial dimensions");
    }
    expectTensor(
        samePadding.forward({1.0f, 2.0f, 3.0f, 4.0f}),
        {1.0f, 2.0f, 3.0f, 4.0f},
        "same padding"
    );

    nncpp::Conv2d fullPadding(1, 1, 2, 2, 5, 1, Padding::Full);
    fullPadding.setWeights(centerKernel);
    fullPadding.setBiases({0.0f});
    if (fullPadding.getOutputWidth() != 6 || fullPadding.getOutputHeight() != 6 ||
        fullPadding.getOutputSize() != 36) {
        throw std::runtime_error("full padding output shape is incorrect");
    }
    nncpp::Tensor expectedFullOutput(36, 0.0f);
    expectedFullOutput[2 * 6 + 2] = 1.0f;
    expectedFullOutput[2 * 6 + 3] = 2.0f;
    expectedFullOutput[3 * 6 + 2] = 3.0f;
    expectedFullOutput[3 * 6 + 3] = 4.0f;
    expectTensor(
        fullPadding.forward({1.0f, 2.0f, 3.0f, 4.0f}),
        expectedFullOutput,
        "full padding"
    );

    return 0;
}
