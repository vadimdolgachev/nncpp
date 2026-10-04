#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <ranges>

namespace nncpp {
    using Scalar = float;
    using Tensor = std::vector<Scalar>;

    class Shape final {
    public:
        constexpr Shape(const std::initializer_list<size_t> dims)
            : rank_(dims.size()),
              total_(dims.size() == 0 ? 0 : 1) {
            if (rank_ > dimensions.size()) {
                throw std::invalid_argument("shape rank is too large");
            }

            size_t index = 0;
            for (const size_t dimension: dims) {
                dimensions[index++] = dimension;
                if (dimension != 0 && total_ > std::numeric_limits<size_t>::max() / dimension) {
                    throw std::overflow_error("shape total size overflows size_t");
                }
                total_ *= dimension;
            }
        }

        [[nodiscard]] constexpr size_t rank() const noexcept {
            return rank_;
        }

        [[nodiscard]] constexpr size_t operator[](const size_t i) const {
            if (i >= rank_) {
                throw std::out_of_range("shape dimension");
            }
            return dimensions[i];
        }

        [[nodiscard]] constexpr size_t total() const noexcept {
            if (rank_ == 0) {
                return 0;
            }
            return total_;
        }

        [[nodiscard]] constexpr bool operator==(const Shape &) const = default;

    private:
        std::array<size_t, 4> dimensions{};
        size_t rank_ = 0;
        size_t total_ = 0;
    };

    /**
     * Interprets a rank-3 Shape using channels-height-width (CHW) order.
     */
    class CHWLayout final {
    public:
        explicit constexpr CHWLayout(const Shape &shape) : shape_(shape) {
            if (shape_.rank() != 3) {
                throw std::invalid_argument("CHW layout requires rank 3");
            }
        }

        [[nodiscard]] constexpr size_t channels() const noexcept {
            return shape_[0];
        }

        [[nodiscard]] constexpr size_t height() const noexcept {
            return shape_[1];
        }

        [[nodiscard]] constexpr size_t width() const noexcept {
            return shape_[2];
        }

        [[nodiscard]] constexpr size_t index(const size_t channel,
                                             const size_t y,
                                             const size_t x) const {
            if (channel >= channels() || y >= height() || x >= width()) {
                throw std::out_of_range("CHW coordinates");
            }
            return (channel * height() + y) * width() + x;
        }

        [[nodiscard]] constexpr const Shape &shape() const noexcept {
            return shape_;
        }

    private:
        Shape shape_;
    };

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor forward(const Tensor &input) = 0;

        virtual Tensor backward(const Tensor &outputGradient) = 0;

        [[nodiscard]] virtual const Shape &getInputShape() const noexcept = 0;

        [[nodiscard]] virtual const Shape &getOutputShape() const noexcept = 0;

        virtual void applyGradient([[maybe_unused]] Scalar learningRate, [[maybe_unused]] size_t actualBatchSize) {
        }

        virtual void resetGradients() {
        }
    };

    class DenseLayer final : public Layer {
    public:
        DenseLayer(const Shape &inputShape_, const Shape &outputShape_);

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

        [[nodiscard]] const Shape &getInputShape() const noexcept override;

        [[nodiscard]] const Shape &getOutputShape() const noexcept override;

    private:
        void initWeights();

        Shape inputShape;
        Shape outputShape;
        Tensor weights;
        Tensor weightGradients;
        Tensor biases;
        Tensor biasGradients;
        Tensor lastInput;
        bool hasForwardResult = false;
    };

    class Sigmoid final : public Layer {
    public:
        explicit Sigmoid(const size_t inputSize) : inputShape({inputSize}),
                                                   output(inputSize) {
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

        [[nodiscard]] const Shape &getInputShape() const noexcept override;

        [[nodiscard]] const Shape &getOutputShape() const noexcept override;

    private:
        Shape inputShape;
        Tensor output;
        bool hasForwardResult = false;
    };

    class ReLU final : public Layer {
    public:
        explicit ReLU(const Shape &inputShape_);

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

        [[nodiscard]] const Shape &getInputShape() const noexcept override;

        [[nodiscard]] const Shape &getOutputShape() const noexcept override;

    private:
        Shape inputShape;
        Tensor output;
        bool hasForwardResult = false;
    };

    class Conv2d final : public Layer {
    public:
        enum class Padding {
            // Do not add zero padding.
            Valid,
            // Preserve spatial dimensions for odd kernels and stride 1.
            Same,
            // Pad every side by kernelSize - 1.
            Full
        };

        Conv2d(const Shape &inputShape_,
               size_t outputChannels_,
               size_t kernelSize_,
               size_t stride_,
               Padding padding_);

        /*
         * output data layout: Channel x Height x Width
         */
        [[nodiscard]] Tensor forward(const Tensor &input) override;

        /**
         * Returns dE/d(input) in CHW order and accumulates weight and bias gradients.
         * Consumes the latest successful forward result; parameters are not updated.
         * Throws std::logic_error without that result or for a mismatched gradient size.
         */
        [[nodiscard]] Tensor backward(const Tensor &outputGradient) override;

        void applyGradient(Scalar learningRate, size_t batchSize) override;

        void resetGradients() override;

        [[nodiscard]] const Tensor &getWeights() const noexcept;

        [[nodiscard]] const Tensor &getWeightGradients() const noexcept;

        [[nodiscard]] const Tensor &getBiases() const noexcept;

        [[nodiscard]] const Tensor &getBiasGradients() const noexcept;

        void setWeights(const Tensor &values);

        void setBiases(const Tensor &values);

        [[nodiscard]] const Shape &getInputShape() const noexcept override;

        [[nodiscard]] const Shape &getOutputShape() const noexcept override;

    private:
        void initWeights();

        [[nodiscard]] size_t calculatePadding() const;

        [[nodiscard]] size_t getWeightIndex(size_t inChannelSize, size_t outChannel, size_t inChannel, size_t ky,
                                            size_t kx) const noexcept;

        Shape inputShape;
        Shape outputShape;
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

    class MaxPool2d final : public Layer {
    public:
        MaxPool2d(const Shape &inputShape_,
                  size_t kernelSize_,
                  size_t stride_);

        [[nodiscard]] Tensor forward(const Tensor &input) override;

        [[nodiscard]] Tensor backward(const Tensor &outputGradient) override;

        [[nodiscard]] const Shape &getInputShape() const noexcept override;

        [[nodiscard]] const Shape &getOutputShape() const noexcept override;

    private:
        Shape inputShape;
        Shape outputShape;
        size_t kernelSize;
        size_t stride;
        Tensor output;
        std::vector<size_t> maxIndices;
        bool hasForwardResult = false;
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
