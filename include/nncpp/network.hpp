#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>
#include <ranges>

namespace nncpp {
    using Scalar = float;
    using Tensor = std::vector<Scalar>;

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor forward(const Tensor &input) = 0;

        virtual Tensor backward(const Tensor &outputGradient) = 0;

        virtual void applyGradient([[maybe_unused]] Scalar learningRate, [[maybe_unused]] size_t actualBatchSize) {
        }

        virtual void resetGradients() {
        }
    };

    class DenseLayer final : public Layer {
    public:
        DenseLayer(std::size_t inputSize_, std::size_t outputSize_);

        [[nodiscard]] Tensor forward(const Tensor &input) override;

        /**
         * Computes the dense layer's backward pass for one sample.
         *
         * `outputGradient[out]` is dE/dy_out. The function stores
         * dE/dW and dE/db in `weightGradients` and `biasGradients`,
         * then returns dE/dx for the preceding layer. `forward()` must
         * be called first so the input used to calculate dE/dW is cached.
         * Gradients accumulate until `resetGradients()` is called.
         */
        [[nodiscard]] Tensor backward(const Tensor &outputGradient) override;

        /**
         * Applies the average of the accumulated gradients.
         *
         * `batchSize` must equal the number of samples whose gradients
         * have been accumulated since the last resetGradients().
         *
         * Gradients are not cleared by this function.
         */
        void applyGradient(Scalar learningRate, size_t batchSize) override;

        void resetGradients() override;

        [[nodiscard]] const Tensor &getWeights() const noexcept;

        [[nodiscard]] const Tensor &getWeightGradients() const noexcept;

        [[nodiscard]] const Tensor &getBiases() const noexcept;

        [[nodiscard]] const Tensor &getBiasGradients() const noexcept;

        void setWeights(const Tensor &values);

        void setBiases(const Tensor &values);

    private:
        void initWeights();

        Tensor weights;
        Tensor weightGradients;
        Tensor biases;
        Tensor biasGradients;
        size_t inputSize;
        size_t outputSize;
        Tensor lastInput;
        bool hasForwardResult = false;
    };

    class Sigmoid final : public Layer {
    public:
        explicit Sigmoid(const size_t inputSize) : output(inputSize) {
        }

        /*
         * input: z
         * return: sigmoid(z)
         */
        [[nodiscard]] Tensor forward(const Tensor &input) override;


        /*
         * input: outputGradient(dE/da)
         * return: dE/dz
         */
        [[nodiscard]] Tensor backward(const Tensor &outputGradient) override;

    private:
        Tensor output;
        bool hasForwardResult = false;
    };

    class ReLU final : public Layer {
    public:
        explicit ReLU(size_t inputSize);

        /*
        * input: z
        * return: ReLU(z)
        */
        [[nodiscard]] Tensor forward(const Tensor &input) override;

        /*
         * input: outputGradient(dE/da)
         * return: dE/dz
         */
        [[nodiscard]] Tensor backward(const Tensor &outputGradient) override;

    private:
        Tensor output;
        bool hasForwardResult = false;
    };

    class Conv2d final : public Layer {
    public:
        enum class Padding {
            Valid,
            Same,
            Full
        };

        Conv2d(const size_t inputChannels_,
               const size_t outputChannels_,
               const size_t width_,
               const size_t height_,
               const size_t kernelSize_,
               const size_t stride_,
               const Padding padding_) : inputChannels(inputChannels_),
                                         outputChannels(outputChannels_),
                                         width(width_),
                                         height(height_),
                                         kernelSize(kernelSize_),
                                         stride(stride_),
                                         padding(padding_) {
            if (inputChannels == 0 || outputChannels == 0
                || width == 0 || height == 0
                || kernelSize == 0
                || stride == 0) {
                throw std::invalid_argument("invalid Conv2d dimensions");
            }
            paddingSize = calculatePadding();
            const auto paddedWidth = width + 2 * paddingSize;
            const auto paddedHeight = height + 2 * paddingSize;
            if (kernelSize > paddedWidth || kernelSize > paddedHeight) {
                throw std::invalid_argument("Conv2d kernel exceeds padded input dimensions");
            }

            outputWidth = (paddedWidth - kernelSize) / stride + 1;
            outputHeight = (paddedHeight - kernelSize) / stride + 1;

            const auto weightsSize = inputChannels * outputChannels * kernelSize * kernelSize;
            weights.resize(weightsSize);
            weightGradients.resize(weightsSize);
            biases.resize(outputChannels);
            biasGradients.resize(outputChannels);
            output.resize(outputChannels * outputHeight * outputWidth);

            initWeights();
        }

        [[nodiscard]] Tensor forward(const Tensor &input) override {
            if (input.size() != inputChannels * width * height) {
                throw std::logic_error("Conv2d input size does not match expected size");
            }
            lastInput = input;

            for (size_t oc = 0; oc < outputChannels; ++oc) {
                for (size_t oy = 0; oy < outputHeight; ++oy) {
                    for (size_t ox = 0; ox < outputWidth; ++ox) {

                        auto sum = biases[oc];

                        for (size_t ic = 0; ic < inputChannels; ++ic) {
                            for (size_t ky = 0; ky < kernelSize; ++ky) {
                                for (size_t kx = 0; kx < kernelSize; ++kx) {
                                    const auto iy = static_cast<std::ptrdiff_t>(oy * stride + ky) -
                                                    static_cast<std::ptrdiff_t>(paddingSize);
                                    const auto ix = static_cast<std::ptrdiff_t>(ox * stride + kx) -
                                                    static_cast<std::ptrdiff_t>(paddingSize);

                                    if (iy < 0
                                        || ix < 0
                                        || static_cast<size_t>(iy) >= height
                                        || static_cast<size_t>(ix) >= width) {
                                        continue;
                                    }

                                    const size_t inputIdx = getInputIndex(
                                        ic,
                                        static_cast<size_t>(iy),
                                        static_cast<size_t>(ix)
                                    );
                                    const size_t weightIdx = getWeightIndex(oc, ic, ky, kx);
                                    sum += weights[weightIdx] * input[inputIdx];
                                }
                            }
                        }

                        output[getOutputIndex(oc, oy, ox)] = sum;
                    }
                }
            }

            hasForwardResult = true;
            return output;
        }

        [[nodiscard]] Tensor backward(const Tensor &outputGradient) override {
            if (!hasForwardResult) {
                throw std::logic_error("forward must be called before backward");
            }
            if (outputGradient.size() != getOutputSize()) {
                throw std::logic_error("Conv2d gradient size does not match output");
            }

            Tensor inputGradient(inputChannels * height * width);

            for (size_t oc = 0; oc < outputChannels; ++oc) {
                for (size_t oy = 0; oy < outputHeight; ++oy) {
                    for (size_t ox = 0; ox < outputWidth; ++ox) {

                        const auto grad = outputGradient[getOutputIndex(oc, oy, ox)];
                        biasGradients[oc] += grad;

                        for (size_t ic = 0; ic < inputChannels; ++ic) {
                            for (size_t ky = 0; ky < kernelSize; ++ky) {
                                for (size_t kx = 0; kx < kernelSize; ++kx) {
                                    const auto iy = static_cast<std::ptrdiff_t>(oy * stride + ky) -
                                                    static_cast<std::ptrdiff_t>(paddingSize);
                                    const auto ix = static_cast<std::ptrdiff_t>(ox * stride + kx) -
                                                    static_cast<std::ptrdiff_t>(paddingSize);

                                    if (iy < 0
                                        || ix < 0
                                        || static_cast<size_t>(iy) >= height
                                        || static_cast<size_t>(ix) >= width) {
                                        continue;
                                    }

                                    const size_t inputIdx = getInputIndex(
                                        ic,
                                        static_cast<size_t>(iy),
                                        static_cast<size_t>(ix)
                                    );
                                    const size_t weightIdx = getWeightIndex(oc, ic, ky, kx);
                                    weightGradients[weightIdx] += lastInput[inputIdx] * grad;
                                    inputGradient[inputIdx] += weights[weightIdx] * grad;
                                }
                            }
                        }
                    }
                }
            }

            hasForwardResult = false;
            return inputGradient;
        }

        void applyGradient(const Scalar learningRate, const size_t batchSize) override {
            if (learningRate <= 0.0f) {
                throw std::invalid_argument("learning rate must be positive");
            }
            if (batchSize == 0) {
                throw std::invalid_argument("batch size must be positive");
            }

            const auto scale = learningRate / static_cast<Scalar>(batchSize);
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] -= scale * weightGradients[i];
            }
            for (size_t i = 0; i < biases.size(); ++i) {
                biases[i] -= scale * biasGradients[i];
            }
        }

        void resetGradients() override {
            std::ranges::fill(weightGradients, 0);
            std::ranges::fill(biasGradients, 0);
        }

        [[nodiscard]] size_t getOutputSize() const noexcept {
            return output.size();
        }

        [[nodiscard]] size_t getOutputWidth() const noexcept {
            return outputWidth;
        }

        [[nodiscard]] size_t getOutputHeight() const noexcept {
            return outputHeight;
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
                throw std::invalid_argument("weights do not match Conv2d dimensions");
            }
            weights = values;
        }

        void setBiases(const Tensor &values) {
            if (values.size() != biases.size()) {
                throw std::invalid_argument("biases do not match Conv2d dimensions");
            }
            biases = values;
        }

    private:
        void initWeights() {
            // Xavier/Glorot initialization for shared convolution kernels.
            const auto kernelArea = static_cast<Scalar>(kernelSize * kernelSize);
            const auto fanSum = static_cast<Scalar>(inputChannels) * kernelArea
                                + static_cast<Scalar>(outputChannels) * kernelArea;
            const auto limit = std::sqrt(6.0f / fanSum);
            thread_local std::mt19937 generator{std::random_device{}()};
            std::uniform_real_distribution distribution(-limit, limit);
            std::ranges::generate(weights, [&distribution] { return distribution(generator); });
        }

        [[nodiscard]] size_t calculatePadding() const {
            if (padding == Padding::Same && (kernelSize % 2 == 0 || stride != 1)) {
                throw std::invalid_argument("Same padding currently requires odd kernel and stride 1");
            }
            switch (padding) {
                case Padding::Valid:
                    return 0;

                case Padding::Same:
                    return (kernelSize - 1) / 2;

                case Padding::Full:
                    return kernelSize - 1;
            }

            throw std::logic_error("invalid padding mode");
        }

        [[nodiscard]] size_t getOutputIndex(const size_t channel,
                                            const size_t y,
                                            const size_t x) const noexcept {
            return channel * outputWidth * outputHeight + y * outputWidth + x;
        }

        [[nodiscard]] size_t getInputIndex(const size_t channel,
                                           const size_t y,
                                           const size_t x) const noexcept {
            return channel * width * height  + y * width + x;
        }

        [[nodiscard]] size_t getWeightIndex(const size_t outChannel,
                                            const size_t inChannel,
                                            const size_t ky,
                                            const size_t kx) const noexcept {
            return ((outChannel * inputChannels + inChannel) * kernelSize + ky) * kernelSize + kx;
        }

        size_t inputChannels;
        size_t outputChannels;
        size_t width;
        size_t height;
        size_t outputWidth;
        size_t outputHeight;
        size_t kernelSize;
        size_t stride;
        Padding padding;
        size_t paddingSize;
        bool hasForwardResult = false;
        Tensor weights;
        Tensor weightGradients;
        Tensor biases;
        Tensor biasGradients;
        Tensor lastInput;
        Tensor output;
    };

    class Loss {
    public:
        virtual ~Loss() = default;

        virtual Scalar forward(const Tensor &input, const Tensor &target) = 0;

        virtual const Tensor &backward() = 0;
    };

    class SoftmaxCategoricalCrossEntropy final : public Loss {
    public:
        explicit SoftmaxCategoricalCrossEntropy(size_t classesSize);

        [[nodiscard]] Scalar forward(const Tensor &logits, const Tensor &target) override;

        [[nodiscard]] const Tensor &backward() override;

    private:
        bool hasForwardResult = false;
        Tensor probabilities;
        Tensor savedTargets;
        Tensor inputGradient;
        Scalar savedTargetSum = 0.0f;
    };

    struct ClassificationMetrics final {
        double averageLoss;
        double accuracy;
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

    Scalar sigmoid(Scalar value);
}
