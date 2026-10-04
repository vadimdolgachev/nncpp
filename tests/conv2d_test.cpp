#include "nncpp/network.hpp"
#include "test_fixture.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
                throw std::runtime_error(label + " contains a non-finite value");
            }
            expectNear(actual[i], expected[i], label);
        }
    }

    struct ConvCase final {
        nncpp::Shape inputShape;
        std::size_t outputChannels;
        std::size_t kernelSize;
        std::size_t stride;
        nncpp::Conv2d::Padding padding;
    };

    std::size_t referencePadding(const ConvCase &config) {
        switch (config.padding) {
            case nncpp::Conv2d::Padding::Valid:
                return 0;
            case nncpp::Conv2d::Padding::Same:
                return (config.kernelSize - 1) / 2;
            case nncpp::Conv2d::Padding::Full:
                return config.kernelSize - 1;
        }
        throw std::runtime_error("unknown test padding");
    }

    nncpp::Shape referenceOutputShape(const ConvCase &config) {
        const auto padding = referencePadding(config);
        return {
            config.outputChannels,
            (config.inputShape[1] + 2 * padding - config.kernelSize) / config.stride + 1,
            (config.inputShape[2] + 2 * padding - config.kernelSize) / config.stride + 1
        };
    }

    std::vector<double> referenceForward(const ConvCase &config,
                                         const nncpp::Tensor &input,
                                         const nncpp::Tensor &weights,
                                         const nncpp::Tensor &biases) {
        const auto channels = config.inputShape[0];
        const auto height = config.inputShape[1];
        const auto width = config.inputShape[2];
        const auto outputShape = referenceOutputShape(config);
        const auto padding = static_cast<std::ptrdiff_t>(referencePadding(config));
        std::vector<double> expected(outputShape.total());

        for (std::size_t och = 0; och < config.outputChannels; ++och) {
            for (std::size_t oy = 0; oy < outputShape[1]; ++oy) {
                for (std::size_t ox = 0; ox < outputShape[2]; ++ox) {
                    double sum = biases[och];
                    for (std::size_t ich = 0; ich < channels; ++ich) {
                        for (std::size_t ky = 0; ky < config.kernelSize; ++ky) {
                            for (std::size_t kx = 0; kx < config.kernelSize; ++kx) {
                                const auto y = static_cast<std::ptrdiff_t>(oy * config.stride + ky) - padding;
                                const auto x = static_cast<std::ptrdiff_t>(ox * config.stride + kx) - padding;
                                if (y < 0 || x < 0 || static_cast<std::size_t>(y) >= height
                                    || static_cast<std::size_t>(x) >= width) {
                                    continue;
                                }
                                const auto inputIndex = (ich * height + static_cast<std::size_t>(y)) * width
                                                        + static_cast<std::size_t>(x);
                                const auto weightIndex = ((och * channels + ich) * config.kernelSize + ky)
                                                         * config.kernelSize + kx;
                                sum += static_cast<double>(input[inputIndex]) * weights[weightIndex];
                            }
                        }
                    }
                    expected[(och * outputShape[1] + oy) * outputShape[2] + ox] = sum;
                }
            }
        }
        return expected;
    }

    void checkForwardCase(const ConvCase &config) {
        nncpp::Conv2d layer(config.inputShape, config.outputChannels, config.kernelSize,
                           config.stride, config.padding);
        if (layer.getOutputShape() != referenceOutputShape(config)) {
            throw std::runtime_error("reference output shape differs from Conv2d");
        }

        nncpp::Tensor input(config.inputShape.total());
        nncpp::Tensor weights(layer.getWeights().size());
        nncpp::Tensor biases(config.outputChannels);
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<nncpp::Scalar>(static_cast<int>(i % 23) - 11) / 13.0f;
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<nncpp::Scalar>(static_cast<int>((i * 7) % 29) - 14) / 17.0f;
        }
        for (std::size_t i = 0; i < biases.size(); ++i) {
            biases[i] = static_cast<nncpp::Scalar>(static_cast<int>(i % 5) - 2) / 5.0f;
        }
        layer.setWeights(weights);
        layer.setBiases(biases);

        const auto label = "Conv2d forward width=" + std::to_string(config.inputShape[2])
                           + " kernel=" + std::to_string(config.kernelSize)
                           + " stride=" + std::to_string(config.stride)
                           + " padding=" + std::to_string(static_cast<int>(config.padding));
        for (std::size_t pass = 0; pass < 3; ++pass) {
            const auto expected = referenceForward(config, input, weights, biases);
            const auto actual = layer.forward(input);
            if (actual.size() != expected.size()) {
                throw std::runtime_error(label + " output size mismatch");
            }
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const double tolerance = 1e-5 + 1e-5 * std::abs(expected[i]);
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])
                    || std::abs(static_cast<double>(actual[i]) - expected[i]) > tolerance) {
                    throw std::runtime_error(label + " pass=" + std::to_string(pass)
                                             + " index=" + std::to_string(i) + " differs from reference");
                }
            }
            for (auto &value : input) {
                value = pass == 0 ? -value * 0.7f + 0.1f : 0.0f;
            }
        }
    }

    void checkForwardBoundaries() {
        using Padding = nncpp::Conv2d::Padding;
        constexpr std::array<std::size_t, 10> interiorWidths = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33};
        for (const auto width : interiorWidths) {
            for (const std::size_t kernel : {1U, 3U, 5U}) {
                for (const auto padding : {Padding::Valid, Padding::Same, Padding::Full}) {
                    checkForwardCase({nncpp::Shape{2, kernel + 2, width + kernel - 1}, 3, kernel, 1, padding});
                }
            }
        }

        for (const auto padding : {Padding::Same, Padding::Full}) {
            checkForwardCase({nncpp::Shape{2, 2, 11}, 3, 5, 1, padding});
            checkForwardCase({nncpp::Shape{2, 7, 2}, 3, 5, 1, padding});
            checkForwardCase({nncpp::Shape{2, 2, 2}, 3, 5, 1, padding});
        }
        for (const auto padding : {Padding::Valid, Padding::Full}) {
            checkForwardCase({nncpp::Shape{3, 9, 32}, 2, 3, 2, padding});
            checkForwardCase({nncpp::Shape{3, 9, 32}, 2, 5, 2, padding});
            checkForwardCase({nncpp::Shape{2, 5, 19}, 3, 2, 1, padding});
        }
        checkForwardCase({nncpp::Shape{1, 28, 28}, 8, 3, 1, Padding::Same});
        checkForwardCase({nncpp::Shape{8, 14, 14}, 16, 3, 1, Padding::Valid});
    }

    struct BackwardReference final {
        std::vector<double> input;
        std::vector<double> weights;
        std::vector<double> biases;
    };

    BackwardReference referenceBackward(const ConvCase &config,
                                        const nncpp::Tensor &input,
                                        const nncpp::Tensor &weights,
                                        const nncpp::Tensor &outputGradient) {
        const auto channels = config.inputShape[0];
        const auto height = config.inputShape[1];
        const auto width = config.inputShape[2];
        const auto outputShape = referenceOutputShape(config);
        const auto padding = static_cast<std::ptrdiff_t>(referencePadding(config));
        BackwardReference result{
            std::vector<double>(input.size()),
            std::vector<double>(weights.size()),
            std::vector<double>(config.outputChannels)
        };

        // Direct unpadded CHW indexing, independent of the production loop order.
        for (std::size_t och = 0; och < config.outputChannels; ++och) {
            for (std::size_t oy = 0; oy < outputShape[1]; ++oy) {
                for (std::size_t ox = 0; ox < outputShape[2]; ++ox) {
                    const double grad = outputGradient[(och * outputShape[1] + oy) * outputShape[2] + ox];
                    result.biases[och] += grad;
                    if (grad == 0.0) {
                        continue;
                    }
                    for (std::size_t ich = 0; ich < channels; ++ich) {
                        for (std::size_t ky = 0; ky < config.kernelSize; ++ky) {
                            for (std::size_t kx = 0; kx < config.kernelSize; ++kx) {
                                const auto y = static_cast<std::ptrdiff_t>(oy * config.stride + ky) - padding;
                                const auto x = static_cast<std::ptrdiff_t>(ox * config.stride + kx) - padding;
                                if (y < 0 || x < 0 || static_cast<std::size_t>(y) >= height
                                    || static_cast<std::size_t>(x) >= width) {
                                    continue;
                                }
                                const auto inputIndex = (ich * height + static_cast<std::size_t>(y)) * width
                                                        + static_cast<std::size_t>(x);
                                const auto weightIndex = ((och * channels + ich) * config.kernelSize + ky)
                                                         * config.kernelSize + kx;
                                result.input[inputIndex] += static_cast<double>(weights[weightIndex]) * grad;
                                result.weights[weightIndex] += static_cast<double>(input[inputIndex]) * grad;
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    void expectReference(const nncpp::Tensor &actual,
                         const std::vector<double> &expected,
                         const std::string &label,
                         const double toleranceScale = 1e-4) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error(label + " size differs from reference");
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double tolerance = toleranceScale * (1.0 + std::abs(expected[i]));
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])
                || std::abs(static_cast<double>(actual[i]) - expected[i]) > tolerance) {
                throw std::runtime_error(label + " index=" + std::to_string(i)
                                         + " actual=" + std::to_string(actual[i])
                                         + " expected=" + std::to_string(expected[i]));
            }
        }
    }

    void checkBackwardCase(const ConvCase &config) {
        nncpp::Conv2d layer(config.inputShape, config.outputChannels, config.kernelSize,
                           config.stride, config.padding);
        nncpp::Tensor input(config.inputShape.total());
        nncpp::Tensor weights(layer.getWeights().size());
        nncpp::Tensor biases(config.outputChannels);
        nncpp::Tensor gradient(referenceOutputShape(config).total());
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<nncpp::Scalar>(static_cast<int>(i % 23) - 11) / 13.0f;
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<nncpp::Scalar>(static_cast<int>((i * 7) % 29) - 14) / 17.0f;
        }
        for (std::size_t i = 0; i < biases.size(); ++i) {
            biases[i] = static_cast<nncpp::Scalar>(static_cast<int>(i % 5) - 2) / 5.0f;
        }
        layer.setWeights(weights);
        layer.setBiases(biases);
        std::vector<double> accumulatedWeights(weights.size());
        std::vector<double> accumulatedBiases(biases.size());
        const auto label = "Conv2d backward outputWidth=" + std::to_string(layer.getOutputShape()[2])
                           + " kernel=" + std::to_string(config.kernelSize)
                           + " stride=" + std::to_string(config.stride)
                           + " padding=" + std::to_string(static_cast<int>(config.padding));

        for (std::size_t pass = 0; pass < 2; ++pass) {
            for (std::size_t i = 0; i < gradient.size(); ++i) {
                auto value = static_cast<nncpp::Scalar>(static_cast<int>((i * 5) % 19) - 9) / 11.0f;
                if (value == 0.0f) {
                    value = 0.125f;
                }
                // Include fully empty blocks, mixed lanes, and negative zero after ReLU.
                gradient[i] = pass == 1 && (i % 32 < 8 || i % 3 == 0)
                              ? (i % 2 == 0 ? 0.0f : -0.0f) : value;
            }
            const auto expected = referenceBackward(config, input, weights, gradient);
            static_cast<void>(layer.forward(input));
            expectThrows<std::logic_error>(
                [&layer] { static_cast<void>(layer.backward({})); }, label + " wrong gradient size");
            expectReference(layer.backward(gradient), expected.input, label + " input pass=" + std::to_string(pass));
            for (std::size_t i = 0; i < weights.size(); ++i) {
                accumulatedWeights[i] += expected.weights[i];
            }
            for (std::size_t i = 0; i < biases.size(); ++i) {
                accumulatedBiases[i] += expected.biases[i];
            }
            expectReference(layer.getWeightGradients(), accumulatedWeights, label + " accumulated weights");
            expectReference(layer.getBiasGradients(), accumulatedBiases, label + " accumulated biases");
            expectTensor(layer.getWeights(), weights, label + " unchanged weights");
            expectTensor(layer.getBiases(), biases, label + " unchanged biases");
            expectThrows<std::logic_error>(
                [&layer, &gradient] { static_cast<void>(layer.backward(gradient)); }, label + " consumed forward");
            for (auto &value : input) {
                value = -value * 0.7f + 0.1f;
            }
        }

        constexpr nncpp::Scalar learningRate = 0.025f;
        layer.applyGradient(learningRate, 2);
        std::vector<double> updatedWeights(weights.size());
        std::vector<double> updatedBiases(biases.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            updatedWeights[i] = weights[i] - static_cast<double>(learningRate / 2.0f) * accumulatedWeights[i];
        }
        for (std::size_t i = 0; i < biases.size(); ++i) {
            updatedBiases[i] = biases[i] - static_cast<double>(learningRate / 2.0f) * accumulatedBiases[i];
        }
        expectReference(layer.getWeights(), updatedWeights, label + " averaged weight update");
        expectReference(layer.getBiases(), updatedBiases, label + " averaged bias update");

        // A zero-gradient sample must not change an already accumulated batch.
        static_cast<void>(layer.forward(input));
        expectTensor(layer.backward(nncpp::Tensor(gradient.size(), 0.0f)),
                     nncpp::Tensor(input.size(), 0.0f), label + " zero input gradient");
        expectReference(layer.getWeightGradients(), accumulatedWeights, label + " zero sample weights");
        expectReference(layer.getBiasGradients(), accumulatedBiases, label + " zero sample biases");
        layer.resetGradients();
        expectTensor(layer.getWeightGradients(), nncpp::Tensor(weights.size(), 0.0f), label + " reset weights");
        expectTensor(layer.getBiasGradients(), nncpp::Tensor(biases.size(), 0.0f), label + " reset biases");
    }

    void checkBackwardBoundaries() {
        using Padding = nncpp::Conv2d::Padding;
        constexpr std::array<std::size_t, 10> outputWidths = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33};
        for (const auto width : outputWidths) {
            for (const std::size_t kernel : {1U, 3U, 5U}) {
                for (const auto padding : {Padding::Valid, Padding::Same, Padding::Full}) {
                    ConvCase config{nncpp::Shape{2, kernel + 2, 1}, 3, kernel, 1, padding};
                    const auto doublePadding = 2 * referencePadding(config);
                    if (width + kernel - 1 <= doublePadding) {
                        continue; // This output width is impossible for Full padding with this kernel.
                    }
                    config.inputShape = {2, kernel + 2, width + kernel - 1 - doublePadding};
                    checkBackwardCase(config);
                }
            }
        }
        for (const auto padding : {Padding::Same, Padding::Full}) {
            checkBackwardCase({nncpp::Shape{2, 2, 11}, 3, 5, 1, padding});
            checkBackwardCase({nncpp::Shape{2, 7, 2}, 3, 5, 1, padding});
            checkBackwardCase({nncpp::Shape{2, 2, 2}, 3, 5, 1, padding});
        }
        for (const auto padding : {Padding::Valid, Padding::Full}) {
            checkBackwardCase({nncpp::Shape{3, 9, 32}, 2, 3, 2, padding});
            checkBackwardCase({nncpp::Shape{3, 9, 32}, 2, 5, 2, padding});
            checkBackwardCase({nncpp::Shape{2, 5, 19}, 3, 2, 1, padding});
        }
        checkBackwardCase({nncpp::Shape{1, 28, 28}, 8, 3, 1, Padding::Same});
        checkBackwardCase({nncpp::Shape{8, 14, 14}, 16, 3, 1, Padding::Valid});
    }

    void checkZeroGradientSkip() {
        nncpp::Conv2d layer(nncpp::Shape{1, 1, 17}, 1, 1, 1, nncpp::Conv2d::Padding::Valid);
        layer.setWeights({1.25f});
        nncpp::Tensor input(17, 0.5f);
        nncpp::Tensor gradient(17, 0.0f);
        for (const std::size_t index : {0U, 8U, 16U}) {
            input[index] = std::numeric_limits<nncpp::Scalar>::quiet_NaN();
        }
        gradient[9] = 2.0f;
        static_cast<void>(layer.forward(input));
        nncpp::Tensor expectedInputGradient(17, 0.0f);
        expectedInputGradient[9] = 2.5f;
        expectTensor(layer.backward(gradient), expectedInputGradient, "zero lanes skip nonfinite input");
        expectTensor(layer.getWeightGradients(), {1.0f}, "zero lanes preserve weight accumulator");
        expectTensor(layer.getBiasGradients(), {2.0f}, "zero lanes bias gradient");

        layer.resetGradients();
        layer.setWeights({std::numeric_limits<nncpp::Scalar>::quiet_NaN()});
        static_cast<void>(layer.forward(input));
        auto inputGradient = layer.backward(gradient);
        if (!std::isnan(inputGradient[9])) {
            throw std::runtime_error("active lane must propagate a nonfinite weight");
        }
        inputGradient[9] = 0.0f;
        expectTensor(inputGradient, nncpp::Tensor(17, 0.0f), "zero lanes skip nonfinite weight");
        expectTensor(layer.getWeightGradients(), {1.0f}, "masked weight gradient remains finite");
        expectTensor(layer.getBiasGradients(), {2.0f}, "masked bias gradient remains finite");
    }

    void checkBackwardFiniteDifferences() {
        nncpp::Conv2d layer(nncpp::Shape{2, 4, 9}, 2, 3, 1, nncpp::Conv2d::Padding::Same);
        nncpp::Tensor input(layer.getInputShape().total());
        nncpp::Tensor weights(layer.getWeights().size());
        nncpp::Tensor biases = {0.1f, -0.2f};
        nncpp::Tensor gradient(layer.getOutputShape().total());
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<nncpp::Scalar>(static_cast<int>(i % 13) - 6) / 9.0f;
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<nncpp::Scalar>(static_cast<int>((i * 3) % 17) - 8) / 11.0f;
        }
        for (std::size_t i = 0; i < gradient.size(); ++i) {
            gradient[i] = static_cast<nncpp::Scalar>(static_cast<int>((i * 7) % 19) - 9) / 13.0f;
        }
        layer.setWeights(weights);
        layer.setBiases(biases);
        static_cast<void>(layer.forward(input));
        const auto inputGradient = layer.backward(gradient);
        const auto weightGradient = layer.getWeightGradients();
        const auto biasGradient = layer.getBiasGradients();

        // J = dot(forward(input), gradient); dJ/d(output) is exactly gradient.
        const auto objective = [&layer, &input, &gradient] {
            const auto output = layer.forward(input);
            double value = 0.0;
            for (std::size_t i = 0; i < output.size(); ++i) {
                value += static_cast<double>(output[i]) * gradient[i];
            }
            return value;
        };
        constexpr nncpp::Scalar epsilon = 1e-3f;
        const auto check = [&objective](nncpp::Tensor &values, const nncpp::Tensor &analytical,
                                       const auto &sync, const std::string &label) {
            std::vector<double> numerical(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                const auto original = values[i];
                values[i] = original + epsilon;
                sync();
                const double plus = objective();
                values[i] = original - epsilon;
                sync();
                const double minus = objective();
                values[i] = original;
                sync();
                numerical[i] = (plus - minus) / (2.0 * static_cast<double>(epsilon));
            }
            expectReference(analytical, numerical, label, 3e-3);
        };
        check(input, inputGradient, [] {}, "finite-difference input gradient");
        check(weights, weightGradient, [&layer, &weights] { layer.setWeights(weights); },
              "finite-difference weight gradient");
        check(biases, biasGradient, [&layer, &biases] { layer.setBiases(biases); },
              "finite-difference bias gradient");
    }
}

int main() {
    checkForwardBoundaries();
    checkBackwardBoundaries();
    checkZeroGradientSkip();
    checkBackwardFiniteDifferences();

    using Padding = nncpp::Conv2d::Padding;

    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(nncpp::Shape{1, 3}, 1, 1, 1, Padding::Valid)); },
        "non-CHW input shape"
    );

    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(nncpp::Shape{0, 3, 3}, 1, 1, 1, Padding::Valid)); },
        "zero input channels"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(nncpp::Shape{1, 3, 3}, 1, 4, 1, Padding::Valid)); },
        "kernel larger than valid input"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(nncpp::Shape{1, 3, 3}, 1, 2, 1, Padding::Same)); },
        "even same-padding kernel"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Conv2d(nncpp::Shape{1, 3, 3}, 1, 3, 2, Padding::Same)); },
        "strided same padding"
    );

    nncpp::Conv2d layer(nncpp::Shape{2, 2, 3}, 1, 2, 1, Padding::Valid);
    if (layer.getInputShape() != nncpp::Shape{2, 2, 3} ||
        layer.getOutputShape() != nncpp::Shape{1, 1, 2}) {
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

    nncpp::Conv2d samePadding(nncpp::Shape{1, 2, 2}, 1, 5, 1, Padding::Same);
    nncpp::Tensor centerKernel(25, 0.0f);
    centerKernel[12] = 1.0f;
    samePadding.setWeights(centerKernel);
    samePadding.setBiases({0.0f});
    if (samePadding.getOutputShape() != nncpp::Shape{1, 2, 2}) {
        throw std::runtime_error("same padding did not preserve spatial dimensions");
    }
    expectTensor(
        samePadding.forward({1.0f, 2.0f, 3.0f, 4.0f}),
        {1.0f, 2.0f, 3.0f, 4.0f},
        "same padding"
    );

    nncpp::Conv2d fullPadding(nncpp::Shape{1, 2, 2}, 1, 5, 1, Padding::Full);
    fullPadding.setWeights(centerKernel);
    fullPadding.setBiases({0.0f});
    if (fullPadding.getOutputShape() != nncpp::Shape{1, 6, 6}) {
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
