#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

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
