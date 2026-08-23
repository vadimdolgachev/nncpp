#include "nncpp/network.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nncpp {
    Scalar sigmoid(const Scalar value) {
        return 1.0f / (1.0f + std::exp(-value));
    }

    DenseLayer::DenseLayer(const std::size_t inputSize_,
                           const std::size_t outputSize_) : weights(inputSize_ * outputSize_),
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

    Tensor DenseLayer::forward(const Tensor &input) {
        if (input.size() != inputSize) {
            throw std::logic_error("input size does not match expected size");
        }
        lastInput = input;
        hasForwardResult = true;
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

    Tensor DenseLayer::backward(const Tensor &outputGradient) {
        if (!hasForwardResult) {
            throw std::logic_error("forward must be called before backward");
        }
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
        hasForwardResult = false;
        return inputGradient;
    }

    void DenseLayer::applyGradient(const Scalar learningRate, const size_t batchSize) {
        if (learningRate <= 0.0) {
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

    void DenseLayer::resetGradients() {
        std::ranges::fill(weightGradients, Scalar{0});
        std::ranges::fill(biasGradients, Scalar{0});
    }

    const Tensor &DenseLayer::getWeights() const noexcept {
        return weights;
    }

    const Tensor &DenseLayer::getWeightGradients() const noexcept {
        return weightGradients;
    }

    const Tensor &DenseLayer::getBiases() const noexcept {
        return biases;
    }

    const Tensor &DenseLayer::getBiasGradients() const noexcept {
        return biasGradients;
    }

    void DenseLayer::setWeights(const Tensor &values) {
        if (values.size() != weights.size()) {
            throw std::invalid_argument("weights do not match layer dimensions");
        }
        weights = values;
    }

    void DenseLayer::setBiases(const Tensor &values) {
        if (values.size() != biases.size()) {
            throw std::invalid_argument("biases do not match layer dimensions");
        }
        biases = values;
    }

    void DenseLayer::initWeights() {
        // Xavier/Glorot initialization
        const auto fanSum = static_cast<Scalar>(inputSize) + static_cast<Scalar>(outputSize);
        const auto limit = std::sqrt(6.0f / fanSum);
        thread_local std::mt19937 generator{std::random_device{}()};
        std::uniform_real_distribution distribution(-limit, limit);
        std::ranges::generate(weights, [&distribution] { return distribution(generator); });
    }

    Tensor Sigmoid::forward(const Tensor &input) {
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

    Tensor Sigmoid::backward(const Tensor &outputGradient) {
        if (!hasForwardResult) {
            throw std::logic_error("forward must be called before backward");
        }
        if (outputGradient.size() != output.size()) {
            throw std::logic_error("output size does not match expected size");
        }
        // dE/dz = dE/da * a(1 - a)
        Tensor inputGradient(outputGradient.size(), 0);
        for (size_t i = 0; i < outputGradient.size(); ++i) {
            inputGradient[i] = outputGradient[i] * output[i] * (1.0f - output[i]);
        }
        hasForwardResult = false;
        return inputGradient;
    }

    ReLU::ReLU(const size_t inputSize) : output(inputSize) {
        if (inputSize == 0) {
            throw std::invalid_argument("ReLU size must be positive");
        }
    }

    Tensor ReLU::forward(const Tensor &input) {
        if (input.size() != output.size()) {
            throw std::logic_error("input size does not match expected size");
        }
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::max(input[i], 0.0f);
        }
        hasForwardResult = true;
        return output;
    }

    Tensor ReLU::backward(const Tensor &outputGradient) {
        if (!hasForwardResult) {
            throw std::logic_error("forward must be called before backward");
        }
        if (outputGradient.size() != output.size()) {
            throw std::logic_error("output size does not match expected size");
        }
        // dE/dz = dE/da * ReLU'
        Tensor inputGradient(outputGradient.size(), 0);
        for (size_t i = 0; i < outputGradient.size(); ++i) {
            inputGradient[i] = output[i] > 0.0f ? outputGradient[i] : 0.0f;
        }
        hasForwardResult = false;
        return inputGradient;
    }

    Conv2d::Conv2d(const size_t inputChannels_,
                   const size_t outputChannels_,
                   const size_t width_,
                   const size_t height_,
                   const size_t kernelSize_, const size_t stride_,
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

    Tensor Conv2d::forward(const Tensor &input) {
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

    Tensor Conv2d::backward(const Tensor &outputGradient) {
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

    void Conv2d::applyGradient(const Scalar learningRate, const size_t batchSize) {
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

    void Conv2d::resetGradients() {
        std::ranges::fill(weightGradients, 0);
        std::ranges::fill(biasGradients, 0);
    }

    size_t Conv2d::getOutputSize() const noexcept {
        return output.size();
    }

    size_t Conv2d::getOutputWidth() const noexcept {
        return outputWidth;
    }

    size_t Conv2d::getOutputHeight() const noexcept {
        return outputHeight;
    }

    const Tensor &Conv2d::getWeights() const noexcept {
        return weights;
    }

    const Tensor &Conv2d::getWeightGradients() const noexcept {
        return weightGradients;
    }

    const Tensor &Conv2d::getBiases() const noexcept {
        return biases;
    }

    const Tensor &Conv2d::getBiasGradients() const noexcept {
        return biasGradients;
    }

    void Conv2d::setWeights(const Tensor &values) {
        if (values.size() != weights.size()) {
            throw std::invalid_argument("weights do not match Conv2d dimensions");
        }
        weights = values;
    }

    void Conv2d::setBiases(const Tensor &values) {
        if (values.size() != biases.size()) {
            throw std::invalid_argument("biases do not match Conv2d dimensions");
        }
        biases = values;
    }

    void Conv2d::initWeights() {
        // Xavier/Glorot initialization for shared convolution kernels.
        const auto kernelArea = static_cast<Scalar>(kernelSize * kernelSize);
        const auto fanSum = static_cast<Scalar>(inputChannels) * kernelArea
                            + static_cast<Scalar>(outputChannels) * kernelArea;
        const auto limit = std::sqrt(6.0f / fanSum);
        thread_local std::mt19937 generator{std::random_device{}()};
        std::uniform_real_distribution distribution(-limit, limit);
        std::ranges::generate(weights, [&distribution] { return distribution(generator); });
    }

    size_t Conv2d::calculatePadding() const {
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

    size_t Conv2d::getOutputIndex(const size_t channel, const size_t y, const size_t x) const noexcept {
        return channel * outputWidth * outputHeight + y * outputWidth + x;
    }

    size_t Conv2d::getInputIndex(const size_t channel, const size_t y, const size_t x) const noexcept {
        return channel * width * height + y * width + x;
    }

    size_t Conv2d::getWeightIndex(const size_t outChannel, const size_t inChannel, const size_t ky,
                                  const size_t kx) const noexcept {
        return ((outChannel * inputChannels + inChannel) * kernelSize + ky) * kernelSize + kx;
    }

    MaxPool2d::MaxPool2d(const size_t channels_,
                         const size_t width_,
                         const size_t height_,
                         const size_t kernelSize_,
                         const size_t stride_) : channels(channels_),
                                                 width(width_),
                                                 height(height_),
                                                 kernelSize(kernelSize_),
                                                 stride(stride_) {
        if (channels == 0) {
            throw std::invalid_argument("channels must be greater than zero");
        }
        if (width == 0 || height == 0) {
            throw std::invalid_argument("width and height must be greater than zero");
        }
        if (kernelSize == 0) {
            throw std::invalid_argument("kernel size must be greater than zero");
        }
        if (kernelSize > width ||
            kernelSize > height) {
            throw std::invalid_argument("kernel is larger than input");
        }
        if (stride == 0) {
            throw std::invalid_argument("stride must be greater than zero");
        }
        outputWidth = (width - kernelSize) / stride + 1;
        outputHeight = (height - kernelSize) / stride + 1;
        output.resize(getOutputSize());
        maxIndices.resize(getOutputSize());
    }

    Tensor MaxPool2d::forward(const Tensor &input) {
        if (input.size() != channels * width * height) {
            throw std::logic_error("MaxPool2d input size does not match");
        }

        for (size_t c = 0; c < channels; c++) {
            for (size_t oy = 0; oy < outputHeight; ++oy) {
                for (size_t ox = 0; ox < outputWidth; ++ox) {
                    const size_t startY = oy * stride;
                    const size_t startX = ox * stride;
                    size_t maxIndex = getInputIndex(c, startY, startX);
                    auto maxValue = input[maxIndex];

                    for (size_t ky = 0; ky < kernelSize; ++ky) {
                        for (size_t kx = 0; kx < kernelSize; ++kx) {
                            const size_t y = startY + ky;
                            const size_t x = startX + kx;
                            if (const size_t inputIndex = getInputIndex(c, y, x); input[inputIndex] > maxValue) {
                                maxValue = input[inputIndex];
                                maxIndex = inputIndex;
                            }
                        }
                    }

                    const size_t outputIndex = getOutputIndex(c, oy, ox);
                    output[outputIndex] = maxValue;
                    maxIndices[outputIndex] = maxIndex;
                }
            }
        }
        hasForwardResult = true;
        return output;
    }

    Tensor MaxPool2d::backward(const Tensor &outputGradient) {
        if (!hasForwardResult) {
            throw std::logic_error("forward must be called before backward");
        }
        if (output.size() != outputGradient.size()) {
            throw std::logic_error("MaxPool2d gradient size does not match");
        }
        Tensor inputGradient(channels * width * height);

        for (size_t i = 0; i < outputGradient.size(); ++i) {
            inputGradient[maxIndices[i]] += outputGradient[i];
        }

        hasForwardResult = false;
        return inputGradient;
    }

    size_t MaxPool2d::getOutputSize() const noexcept {
        return outputWidth * outputHeight * channels;
    }

    size_t MaxPool2d::getOutputIndex(const size_t channel, const size_t y, const size_t x) const noexcept {
        return channel * outputWidth * outputHeight + y * outputWidth + x;
    }

    size_t MaxPool2d::getInputIndex(const size_t channel, const size_t y, const size_t x) const noexcept {
        return channel * width * height + y * width + x;
    }

    SoftmaxCategoricalCrossEntropy::SoftmaxCategoricalCrossEntropy(const size_t classesSize) : probabilities(
            classesSize),
        inputGradient(classesSize) {
        if (classesSize == 0) {
            throw std::invalid_argument("class count must be positive");
        }
    }

    Scalar SoftmaxCategoricalCrossEntropy::forward(const Tensor &logits, const Tensor &target) {
        if (logits.size() != probabilities.size()) {
            throw std::invalid_argument("logit size does not match class count");
        }
        if (target.size() != probabilities.size()) {
            throw std::invalid_argument("target size does not match class count");
        }

        const auto maxElem = *std::ranges::max_element(logits);
        if (!std::isfinite(maxElem)) {
            throw std::invalid_argument("logits must be finite");
        }

        Scalar sumExp = 0.0;
        Scalar targetSum = 0.0;
        Scalar targetDotShiftedLogits = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (!std::isfinite(logits[i])) {
                throw std::invalid_argument("logits must be finite");
            }
            if (!std::isfinite(target[i]) || target[i] < 0.0f) {
                throw std::invalid_argument("targets must be finite and nonnegative");
            }
            probabilities[i] = std::exp(logits[i] - maxElem);
            sumExp += probabilities[i];
            targetSum += target[i];
            targetDotShiftedLogits += target[i] * (logits[i] - maxElem);
        }
        if (constexpr Scalar targetTolerance = 1e-5f; std::abs(targetSum - 1.0f) > targetTolerance) {
            throw std::invalid_argument("targets must sum to one");
        }

        std::ranges::transform(probabilities, probabilities.begin(), [sumExp](auto p) {
            return p / sumExp;
        });

        hasForwardResult = true;
        savedTargets = target;
        savedTargetSum = targetSum;

        return targetSum * std::log(sumExp) - targetDotShiftedLogits;
    }

    const Tensor &SoftmaxCategoricalCrossEntropy::backward() {
        if (!hasForwardResult) {
            throw std::logic_error("forward must be called before backward");
        }
        std::ranges::transform(probabilities, savedTargets, inputGradient.begin(),
                               [](const auto p, const auto t) { return p - t; });
        hasForwardResult = false;
        return inputGradient;
    }

    Scalar derivativeSigmoid(const Scalar activation) {
        return activation * (1.0f - activation);
    }

    Scalar MSE(const Scalar output, const Scalar target) {
        const Scalar difference = output - target;
        return 0.5f * difference * difference;
    }

    Tensor MSE(const Tensor &output, const Tensor &target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("output do not match output layer");
        }
        Tensor outputVector(output.size());
        for (std::size_t i = 0; i < output.size(); ++i) {
            outputVector[i] = MSE(output[i], target[i]);
        }
        return outputVector;
    }

    Tensor derivativeMSE(const Tensor &output, const Tensor &target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("output and target sizes differ");
        }

        Tensor gradient(output.size());
        for (std::size_t i = 0; i < output.size(); ++i) {
            gradient[i] = output[i] - target[i];
        }
        return gradient;
    }

    Scalar outputDelta(const Scalar output, const Scalar target) {
        return (output - target) * derivativeSigmoid(output);
    }

    void forwardPass(std::vector<std::vector<Neuron> > &network) {
        if (network.empty()) {
            throw std::invalid_argument("network must not be empty");
        }

        for (std::size_t layerIndex = 1; layerIndex < network.size(); ++layerIndex) {
            const auto &previousLayer = network[layerIndex - 1];

            for (Neuron &neuron: network[layerIndex]) {
                if (neuron.weights.size() != previousLayer.size()) {
                    throw std::logic_error("neuron weights do not match previous layer");
                }

                auto z = neuron.bias;
                for (std::size_t weightIndex = 0; weightIndex < neuron.weights.size(); ++weightIndex) {
                    z += neuron.weights[weightIndex] * previousLayer[weightIndex].activation;
                }
                neuron.activation = sigmoid(z);
            }
        }
    }

    void calculateDeltas(std::vector<std::vector<Neuron> > &network, const Tensor &targets) {
        if (network.size() < 2) {
            throw std::invalid_argument("network must contain at least two layers");
        }
        if (targets.size() != network.back().size()) {
            throw std::invalid_argument("targets do not match output layer");
        }

        for (std::size_t neuronIndex = 0; neuronIndex < network.back().size(); ++neuronIndex) {
            Neuron &neuron = network.back()[neuronIndex];
            neuron.delta = outputDelta(neuron.activation, targets[neuronIndex]);
        }

        for (std::size_t layerIndex = network.size() - 2; layerIndex > 0; --layerIndex) {
            const auto &nextLayer = network[layerIndex + 1];
            const auto currentLayerSize = network[layerIndex].size();

            for (const Neuron &nextNeuron: nextLayer) {
                if (nextNeuron.weights.size() != currentLayerSize) {
                    throw std::logic_error("neuron weights do not match previous layer");
                }
            }

            for (std::size_t neuronIndex = 0; neuronIndex < currentLayerSize; ++neuronIndex) {
                Scalar dE_da = 0.0f;
                for (const Neuron &nextNeuron: nextLayer) {
                    dE_da += nextNeuron.delta * nextNeuron.weights[neuronIndex];
                }

                Neuron &neuron = network[layerIndex][neuronIndex];
                neuron.delta = dE_da * derivativeSigmoid(neuron.activation);
            }
        }
    }

    void optimizeParams(std::vector<std::vector<Neuron> > &network, const Scalar learningRate) {
        if (network.size() < 2) {
            throw std::invalid_argument("network must contain at least two layers");
        }
        if (learningRate <= 0.0) {
            throw std::invalid_argument("learning rate must be positive");
        }
        for (std::size_t layerIndex = 1; layerIndex < network.size(); ++layerIndex) {
            const auto &prevLayer = network[layerIndex - 1];

            for (Neuron &neuron: network[layerIndex]) {
                if (neuron.weights.size() != prevLayer.size()) {
                    throw std::logic_error("neuron weights do not match previous layer");
                }

                for (std::size_t weightIndex = 0; weightIndex < neuron.weights.size(); ++weightIndex) {
                    const Scalar dE_dw = neuron.delta * prevLayer[weightIndex].activation;
                    neuron.weights[weightIndex] -= learningRate * dE_dw;
                }
                const Scalar dE_db = neuron.delta;
                neuron.bias -= learningRate * dE_db;
            }
        }
    }

    Scalar loss(const Tensor &output, const Tensor &targets) {
        if (output.size() != targets.size()) {
            throw std::invalid_argument("output and target sizes differ");
        }

        Scalar total = 0.0;
        for (std::size_t index = 0; index < output.size(); ++index) {
            total += MSE(output[index], targets[index]);
        }
        return total;
    }

    Scalar loss(const std::vector<Neuron> &outputs, const Tensor &targets) {
        if (outputs.size() != targets.size()) {
            throw std::invalid_argument("targets do not match output layer");
        }

        Scalar total = 0.0;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            total += MSE(outputs[index].activation, targets[index]);
        }
        return total;
    }

    Tensor &sigmoid(Tensor &input) {
        for (Scalar &index: input) {
            index = sigmoid(index);
        }
        return input;
    }
}
