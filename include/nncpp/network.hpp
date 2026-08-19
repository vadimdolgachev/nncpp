#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace nncpp {
    using Scalar = float;
    using Tensor = std::vector<Scalar>;

    Scalar sigmoid(Scalar value);

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor forward(const Tensor &input) = 0;

        virtual Tensor backward(const Tensor &outputGradient) = 0;

        virtual void applyGradient(Scalar) {
        }

        virtual void resetGradients() {
        }
    };

    class DenseLayer final : public Layer {
    public:
        DenseLayer(const std::size_t inputSize_,
                   const std::size_t outputSize_)
            : weights(inputSize_ * outputSize_),
              weightGradients(inputSize_ * outputSize_, 0),
              biases(outputSize_, 0),
              biasGradients(outputSize_, 0),
              inputSize(inputSize_),
              outputSize(outputSize_) {
            if (inputSize == 0 || outputSize == 0) {
                throw std::invalid_argument("dense layer dimensions must be positive");
            }

            initWeights();
        }

        [[nodiscard]] Tensor forward(const Tensor &input) override {
            if (input.size() != inputSize) {
                throw std::logic_error("input size does not match expected size");
            }
            lastInput = input;
            Tensor output(outputSize, 0);
            // z = W * x + b
            for (size_t outIndex = 0; outIndex < outputSize; ++outIndex) {
                Scalar z = biases[outIndex];
                for (size_t inIndex = 0; inIndex < inputSize; ++inIndex) {
                    const size_t wIndex = inIndex + inputSize * outIndex;
                    z += weights[wIndex] * input[inIndex];
                }
                output[outIndex] = z;
            }
            return output;
        }

        /**
         * Computes the dense layer's backward pass for one sample.
         *
         * `outputGradient[out]` is dE/dy_out. The function stores
         * dE/dW and dE/db in `weightGradients` and `biasGradients`,
         * then returns dE/dx for the preceding layer. `forward()` must
         * be called first so the input used to calculate dE/dW is cached.
         * Gradients accumulate until `resetGradients()` is called.
         */
        [[nodiscard]] Tensor backward(const Tensor &outputGradient) override {
            if (outputGradient.size() != outputSize) {
                throw std::logic_error("gradient size does not match output size");
            }
            if (lastInput.size() != inputSize) {
                throw std::logic_error("forward must be called before backward");
            }

            Tensor inputGradient(inputSize, 0);

            for (size_t outIndex = 0; outIndex < outputSize; ++outIndex) {
                const auto grad = outputGradient[outIndex];
                biasGradients[outIndex] += grad;
                for (size_t inIndex = 0; inIndex < inputSize; ++inIndex) {
                    const size_t wIndex = inIndex + inputSize * outIndex;
                    weightGradients[wIndex] += lastInput[inIndex] * grad;
                    inputGradient[inIndex] += weights[wIndex] * grad;
                }
            }
            return inputGradient;
        }

        /**
         * Applies the accumulated gradients without clearing them.
         * Call resetGradients() explicitly before accumulating a new batch.
         */
        void applyGradient(const Scalar learningRate) override {
            if (learningRate <= 0.0) {
                throw std::invalid_argument("learning rate must be positive");
            }
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] -= learningRate * weightGradients[i];
            }
            for (size_t i = 0; i < biases.size(); ++i) {
                biases[i] -= learningRate * biasGradients[i];
            }
        }

        void resetGradients() override {
            std::ranges::fill(weightGradients, Scalar{0});
            std::ranges::fill(biasGradients, Scalar{0});
        }

        [[nodiscard]] const Tensor &getWeights() const noexcept {
            return weights;
        }

        [[nodiscard]] const Tensor &getWeightGradients() const noexcept {
            return weightGradients;
        }

        [[nodiscard]] const Tensor &getBiases() const noexcept {
            return biases;
        }

        [[nodiscard]] const Tensor &getBiasGradients() const noexcept {
            return biasGradients;
        }

        void setWeights(const Tensor &values) {
            if (values.size() != weights.size()) {
                throw std::invalid_argument("weights do not match layer dimensions");
            }
            weights = values;
        }

        void setBiases(const Tensor &values) {
            if (values.size() != biases.size()) {
                throw std::invalid_argument("biases do not match layer dimensions");
            }
            biases = values;
        }

    private:
        void initWeights() {
            // Xavier/Glorot initialization
            const Scalar fanSum = static_cast<Scalar>(inputSize) + static_cast<Scalar>(outputSize);
            const Scalar limit = std::sqrt(Scalar{6} / fanSum);
            thread_local std::mt19937 generator{std::random_device{}()};
            std::uniform_real_distribution<Scalar> distribution(-limit, limit);
            std::ranges::generate(weights, [&distribution] { return distribution(generator); });
        }

        Tensor weights;
        Tensor weightGradients;
        Tensor biases;
        Tensor biasGradients;
        size_t inputSize;
        size_t outputSize;
        Tensor lastInput;
    };

    class Sigmoid final : public Layer {
    public:
        explicit Sigmoid(const size_t inputSize) : output(inputSize) {
        }

        /*
         * input: z
         * return: sigmoid(z)
         */
        [[nodiscard]] Tensor forward(const Tensor &input) override {
            if (input.size() != output.size()) {
                throw std::logic_error("input size does not match expected size");
            }
            // a = sigmoid(z)
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = sigmoid(input[i]);
            }
            hasForwardResult = true;
            return output;
        }


        /*
         * input: outputGradient(dE/da)
         * return: dE/dz
         */
        [[nodiscard]] Tensor backward(const Tensor &outputGradient) override {
            if (!hasForwardResult) {
                throw std::logic_error("forward must be called before backward");
            }
            if (outputGradient.size() != output.size()) {
                throw std::logic_error("output size does not match expected size");
            }
            // dE/dz = dE/da * a(1 - a)
            Tensor inputGradient(outputGradient.size(), Scalar{0});
            for (size_t i = 0; i < outputGradient.size(); ++i) {
                inputGradient[i] = outputGradient[i] * output[i] * (Scalar{1} - output[i]);
            }
            return inputGradient;
        }

    private:
        Tensor output;
        bool hasForwardResult = false;
    };

    struct Neuron final {
        Tensor weights;
        Scalar bias = 0.0f;
        Scalar activation = 0.0f;
        Scalar delta = 0.0f;
    };

    Scalar derivativeSigmoid(Scalar activation);

    Scalar MSE(Scalar output, Scalar target);

    Tensor MSE(const Tensor &output, const Tensor &target);

    Tensor derivativeMSE(const Tensor &output, const Tensor &target);

    Scalar outputDelta(Scalar output, Scalar target);

    void forwardPass(std::vector<std::vector<Neuron> > &network);

    void calculateDeltas(std::vector<std::vector<Neuron> > &network, const Tensor &targets);

    void optimizeParams(std::vector<std::vector<Neuron> > &network, Scalar learningRate);

    Scalar loss(const Tensor &output, const Tensor &targets);

    Scalar loss(const std::vector<Neuron> &outputs, const Tensor &targets);

    Tensor &sigmoid(Tensor &input);
}
