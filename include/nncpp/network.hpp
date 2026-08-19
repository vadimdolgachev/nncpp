#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace nncpp {
    double sigmoid(double value);

    class DenseLayer final {
    public:
        DenseLayer(const std::size_t inputSize_,
                   const std::size_t outputSize_)
            : weights(inputSize_ * outputSize_),
              weightGradients(inputSize_ * outputSize_, 0.0),
              biases(outputSize_, 0.0),
              biasGradients(outputSize_, 0.0),
              inputSize(inputSize_),
              outputSize(outputSize_) {
            if (inputSize == 0 || outputSize == 0) {
                throw std::invalid_argument("dense layer dimensions must be positive");
            }

            initWeights();
        }

        [[nodiscard]] std::vector<double> forward(const std::vector<double> &input) {
            if (input.size() != inputSize) {
                throw std::logic_error("input size does not match expected size");
            }
            lastInput = input;
            std::vector output(outputSize, 0.0);
            // z = W * x + b
            for (size_t outIndex = 0; outIndex < outputSize; ++outIndex) {
                double z = biases[outIndex];
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
        [[nodiscard]] std::vector<double> backward(const std::vector<double> &outputGradient) {
            if (outputGradient.size() != outputSize) {
                throw std::logic_error("gradient size does not match output size");
            }
            if (lastInput.size() != inputSize) {
                throw std::logic_error("forward must be called before backward");
            }

            std::vector inputGradient(inputSize, 0.0);

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
        void applyGradient(const double learningRate) {
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

        void resetGradients() {
            std::ranges::fill(weightGradients, 0.0);
            std::ranges::fill(biasGradients, 0.0);
        }

        [[nodiscard]] const std::vector<double> &getWeights() const noexcept {
            return weights;
        }

        [[nodiscard]] const std::vector<double> &getWeightGradients() const noexcept {
            return weightGradients;
        }

        [[nodiscard]] const std::vector<double> &getBiases() const noexcept {
            return biases;
        }

        [[nodiscard]] const std::vector<double> &getBiasGradients() const noexcept {
            return biasGradients;
        }

        void setWeights(const std::vector<double> &values) {
            if (values.size() != weights.size()) {
                throw std::invalid_argument("weights do not match layer dimensions");
            }
            weights = values;
        }

        void setBiases(const std::vector<double> &values) {
            if (values.size() != biases.size()) {
                throw std::invalid_argument("biases do not match layer dimensions");
            }
            biases = values;
        }

    private:
        void initWeights() {
            // Xavier/Glorot initialization
            const double fanSum = static_cast<double>(inputSize) + static_cast<double>(outputSize);
            const double limit = std::sqrt(6.0 / fanSum);
            static std::random_device rd{};
            static std::mt19937 gen{rd()};
            std::uniform_real_distribution dist(-limit, limit);
            std::ranges::generate(weights, [&dist] { return dist(gen); });
        }

        std::vector<double> weights;
        std::vector<double> weightGradients;
        std::vector<double> biases;
        std::vector<double> biasGradients;
        size_t inputSize;
        size_t outputSize;
        std::vector<double> lastInput;
    };

    class Sigmoid final {
    public:
        explicit Sigmoid(const size_t inputSize) : output(inputSize) {
        }

        /*
         * input: z
         * return: sigmoid(z)
         */
        [[nodiscard]] std::vector<double> forward(const std::vector<double> &input) {
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
        [[nodiscard]] std::vector<double> backward(const std::vector<double> &outputGradient) const {
            if (!hasForwardResult) {
                throw std::logic_error("forward must be called before backward");
            }
            if (outputGradient.size() != output.size()) {
                throw std::logic_error("output size does not match expected size");
            }
            // dE/dz = dE/da * a(1 - a)
            std::vector inputGradient(outputGradient.size(), 0.0);
            for (size_t i = 0; i < outputGradient.size(); ++i) {
                inputGradient[i] = outputGradient[i] * output[i] * (1.0 - output[i]);
            }
            return inputGradient;
        }

    private:
        std::vector<double> output;
        bool hasForwardResult = false;
    };

    struct Neuron final {
        std::vector<double> weights;
        double bias = 0.0;
        double activation = 0.0;
        double delta = 0.0;
    };

    using Layer = std::vector<Neuron>;
    using Network = std::vector<Layer>;

    double derivativeSigmoid(double activation);

    double MSE(double output, double target);

    std::vector<double> MSE(const std::vector<double> &output, const std::vector<double> &target);

    std::vector<double> derivativeMSE(const std::vector<double> &output, const std::vector<double> &target);

    double outputDelta(double output, double target);

    void forwardPass(Network &network);

    void calculateDeltas(Network &network, const std::vector<double> &targets);

    void optimizeParams(Network &network, double learningRate);

    double loss(const std::vector<double> &output, const std::vector<double> &targets);

    double loss(const Layer &outputs, const std::vector<double> &targets);

    std::vector<double> &sigmoid(std::vector<double> &input);
}
