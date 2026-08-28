#include "nncpp/network.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nncpp {
    Scalar sigmoid(const Scalar value) {
        return 1.0f / (1.0f + std::exp(-value));
    }

    DenseLayer::DenseLayer(const Shape &inputShape_,
                           const Shape &outputShape_) : inputShape(inputShape_),
                                                        outputShape(outputShape_),
                                                        weights(inputShape_.total() * outputShape_.total()),
                                                        weightGradients(weights.size(), 0),
                                                        biases(outputShape_.total(), 0),
                                                        biasGradients(outputShape_.total(), 0) {
        if (inputShape.total() == 0 || outputShape.total() == 0) {
            throw std::invalid_argument("dense layer dimensions must be positive");
        }

        initWeights();
    }

    Tensor DenseLayer::forward(const Tensor &input) {
        if (input.size() != inputShape.total()) {
            throw std::logic_error("input size does not match expected size");
        }
        lastInput = input;
        hasForwardResult = true;

        // z = W * x + b
        const size_t inTotal = inputShape.total();
        const size_t outTotal = outputShape.total();
        Tensor output(outTotal, 0);
#if defined(__AVX2__)
        for (size_t outIndex = 0; outIndex < outTotal; ++outIndex) {
            const Scalar z = biases[outIndex];
            const Scalar *W = weights.data() + inTotal * outIndex;
            const Scalar *x = input.data();

            __m256 m_sum1 = _mm256_setzero_ps();
            __m256 m_sum2 = _mm256_setzero_ps();

            size_t i = 0;
            for (; i + 15 < inTotal; i += 16) {
                m_sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(&W[i]), _mm256_loadu_ps(&x[i]), m_sum1);
                m_sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(&W[i + 8]), _mm256_loadu_ps(&x[i + 8]), m_sum2);
            }

            m_sum1 = _mm256_add_ps(m_sum1, m_sum2);

            for (; i + 7 < inTotal; i += 8) {
                m_sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(&W[i]), _mm256_loadu_ps(&x[i]), m_sum1);
            }

            __m128 xlow = _mm256_castps256_ps128(m_sum1);
            __m128 xhigh = _mm256_extractf128_ps(m_sum1, 1);
            xlow = _mm_add_ps(xlow, xhigh);
            __m128 xshuf = _mm_movehl_ps(xlow, xlow);
            xlow = _mm_add_ps(xlow, xshuf);
            xshuf = _mm_shuffle_ps(xlow, xlow, 1);
            xlow = _mm_add_ps(xlow, xshuf);

            float zSimd = _mm_cvtss_f32(xlow);

            for (; i < inTotal; ++i) {
                zSimd += W[i] * x[i];
            }

            output[outIndex] = z + zSimd;
        }
#else
        for (size_t outIndex = 0; outIndex < outputShape.total(); ++outIndex) {
            Scalar z = biases[outIndex];
            for (size_t inIndex = 0; inIndex < inputShape.total(); ++inIndex) {
                const size_t wIndex = inIndex + inputShape.total() * outIndex;
                z += weights[wIndex] * input[inIndex];
            }
            output[outIndex] = z;
        }
#endif
        return output;
    }

    Tensor DenseLayer::backward(const Tensor &outputGradient) {
        if (!hasForwardResult) {
            throw std::logic_error("forward must be called before backward");
        }
        if (outputGradient.size() != outputShape.total()) {
            throw std::logic_error("gradient size does not match output size");
        }
        if (lastInput.size() != inputShape.total()) {
            throw std::logic_error("forward must be called before backward");
        }

        Tensor inputGradient(inputShape.total(), 0);

        for (size_t outIndex = 0; outIndex < outputShape.total(); ++outIndex) {
            const auto grad = outputGradient[outIndex];
            biasGradients[outIndex] += grad;
            for (size_t inIndex = 0; inIndex < inputShape.total(); ++inIndex) {
                const size_t wIndex = inIndex + inputShape.total() * outIndex;
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

    const Shape &DenseLayer::getInputShape() const noexcept {
        return inputShape;
    }

    const Shape &DenseLayer::getOutputShape() const noexcept {
        return outputShape;
    }

    void DenseLayer::initWeights() {
        // Xavier/Glorot initialization
        const auto fanSum = static_cast<Scalar>(inputShape.total()) + static_cast<Scalar>(outputShape.total());
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

    const Shape &Sigmoid::getInputShape() const noexcept {
        return inputShape;
    }

    const Shape &Sigmoid::getOutputShape() const noexcept {
        return inputShape;
    }

    ReLU::ReLU(const Shape &inputShape_) : inputShape(inputShape_),
                                           output(inputShape.total()) {
        if (inputShape.total() == 0) {
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

    const Shape &ReLU::getInputShape() const noexcept {
        return inputShape;
    }

    const Shape &ReLU::getOutputShape() const noexcept {
        return inputShape;
    }

    Conv2d::Conv2d(const Shape &inputShape_,
                   const size_t outputChannels_,
                   const size_t kernelSize_,
                   const size_t stride_,
                   const Padding padding_) : inputShape(inputShape_),
                                             outputShape({}),
                                             kernelSize(kernelSize_),
                                             stride(stride_),
                                             padding(padding_) {
        const CHWLayout inputLayout(inputShape);
        if (inputLayout.channels() == 0 || outputChannels_ == 0
            || inputLayout.width() == 0 || inputLayout.height() == 0
            || kernelSize_ == 0
            || stride_ == 0) {
            throw std::invalid_argument("invalid Conv2d dimensions");
        }
        paddingSize = calculatePadding();
        const auto paddedWidth = inputLayout.width() + 2 * paddingSize;
        const auto paddedHeight = inputLayout.height() + 2 * paddingSize;
        if (kernelSize > paddedWidth || kernelSize > paddedHeight) {
            throw std::invalid_argument("Conv2d kernel exceeds padded input dimensions");
        }

        // CHW layout.
        outputShape = {
            outputChannels_,
            (paddedHeight - kernelSize) / stride + 1,
            (paddedWidth - kernelSize) / stride + 1
        };

        const auto weightsSize = inputLayout.channels() * outputChannels_ * kernelSize * kernelSize;
        weights.resize(weightsSize);
        weightGradients.resize(weightsSize);
        biases.resize(outputChannels_);
        biasGradients.resize(outputChannels_);
        output.resize(outputShape.total());

        initWeights();
    }

    Tensor Conv2d::forward(const Tensor &input) {
        if (input.size() != inputShape.total()) {
            throw std::logic_error("Conv2d input size does not match expected size");
        }
        lastInput = input;

        const CHWLayout outputLayout(outputShape);
        const CHWLayout inputLayout(inputShape);
        const auto oChannels = outputLayout.channels();
        const auto oWidth = outputLayout.width();
        const auto oHeight = outputLayout.height();
        const auto iChannels = inputLayout.channels();
        const auto iWidth = inputLayout.width();
        const auto iHeight = inputLayout.height();

        for (size_t och = 0; och < oChannels; ++och) {
            for (size_t oy = 0; oy < oHeight; ++oy) {
                for (size_t ox = 0; ox < oWidth; ++ox) {
                    auto sum = biases[och];

                    for (size_t ich = 0; ich < iChannels; ++ich) {
                        for (size_t ky = 0; ky < kernelSize; ++ky) {
                            for (size_t kx = 0; kx < kernelSize; ++kx) {
                                const auto iy = static_cast<std::ptrdiff_t>(oy * stride + ky) -
                                                static_cast<std::ptrdiff_t>(paddingSize);
                                const auto ix = static_cast<std::ptrdiff_t>(ox * stride + kx) -
                                                static_cast<std::ptrdiff_t>(paddingSize);

                                if (iy < 0
                                    || ix < 0
                                    || static_cast<size_t>(iy) >= iHeight
                                    || static_cast<size_t>(ix) >= iWidth) {
                                    continue;
                                }

                                const size_t inputIdx = inputLayout.index(
                                    ich, static_cast<size_t>(iy), static_cast<size_t>(ix));
                                const size_t weightIdx = getWeightIndex(iChannels, och, ich, ky, kx);
                                sum += weights[weightIdx] * input[inputIdx];
                            }
                        }
                    }

                    output[outputLayout.index(och, oy, ox)] = sum;
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
        if (outputGradient.size() != outputShape.total()) {
            throw std::logic_error("Conv2d gradient size does not match output");
        }

        Tensor inputGradient(inputShape.total(), Scalar{0});

        const CHWLayout outputLayout(outputShape);
        const CHWLayout inputLayout(inputShape);
        const auto oChannels = outputLayout.channels();
        const auto oWidth = outputLayout.width();
        const auto oHeight = outputLayout.height();
        const auto iChannels = inputLayout.channels();
        const auto iWidth = inputLayout.width();
        const auto iHeight = inputLayout.height();

        for (size_t och = 0; och < oChannels; ++och) {
            for (size_t oy = 0; oy < oHeight; ++oy) {
                for (size_t ox = 0; ox < oWidth; ++ox) {
                    const auto grad = outputGradient[outputLayout.index(och, oy, ox)];
                    biasGradients[och] += grad;

                    for (size_t ich = 0; ich < iChannels; ++ich) {
                        for (size_t ky = 0; ky < kernelSize; ++ky) {
                            for (size_t kx = 0; kx < kernelSize; ++kx) {
                                const auto iy = static_cast<std::ptrdiff_t>(oy * stride + ky) -
                                                static_cast<std::ptrdiff_t>(paddingSize);
                                const auto ix = static_cast<std::ptrdiff_t>(ox * stride + kx) -
                                                static_cast<std::ptrdiff_t>(paddingSize);

                                if (iy < 0
                                    || ix < 0
                                    || static_cast<size_t>(iy) >= iHeight
                                    || static_cast<size_t>(ix) >= iWidth) {
                                    continue;
                                }

                                const size_t inputIdx = inputLayout.index(
                                    ich, static_cast<size_t>(iy), static_cast<size_t>(ix));
                                const size_t weightIdx = getWeightIndex(iChannels, och, ich, ky, kx);
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

    const Shape &Conv2d::getInputShape() const noexcept {
        return inputShape;
    }

    const Shape &Conv2d::getOutputShape() const noexcept {
        return outputShape;
    }

    void Conv2d::initWeights() {
        // Xavier/Glorot initialization for shared convolution kernels.
        const CHWLayout inputLayout(inputShape);
        const CHWLayout outputLayout(outputShape);
        const auto kernelArea = static_cast<Scalar>(kernelSize * kernelSize);
        const auto fanSum = static_cast<Scalar>(inputLayout.channels()) * kernelArea
                            + static_cast<Scalar>(outputLayout.channels()) * kernelArea;
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

    size_t Conv2d::getWeightIndex(const size_t inChannelSize,
                                  const size_t outChannel,
                                  const size_t inChannel,
                                  const size_t ky,
                                  const size_t kx) const noexcept {
        return ((outChannel * inChannelSize + inChannel) * kernelSize + ky) * kernelSize + kx;
    }

    MaxPool2d::MaxPool2d(const Shape &inputShape_,
                         const size_t kernelSize_,
                         const size_t stride_) : inputShape(inputShape_),
                                                 outputShape({}),
                                                 kernelSize(kernelSize_),
                                                 stride(stride_) {
        const CHWLayout inputLayout(inputShape);
        if (inputLayout.channels() == 0) {
            throw std::invalid_argument("channels must be greater than zero");
        }
        if (inputLayout.width() == 0 || inputLayout.height() == 0) {
            throw std::invalid_argument("width and height must be greater than zero");
        }
        if (kernelSize == 0) {
            throw std::invalid_argument("kernel size must be greater than zero");
        }
        if (kernelSize > inputLayout.width() || kernelSize > inputLayout.height()) {
            throw std::invalid_argument("kernel is larger than input");
        }
        if (stride == 0) {
            throw std::invalid_argument("stride must be greater than zero");
        }
        const auto outputWidth = (inputLayout.width() - kernelSize) / stride + 1;
        const auto outputHeight = (inputLayout.height() - kernelSize) / stride + 1;
        outputShape = {inputLayout.channels(), outputHeight, outputWidth};
        output.resize(outputShape.total());
        maxIndices.resize(outputShape.total());
    }

    Tensor MaxPool2d::forward(const Tensor &input) {
        if (input.size() != inputShape.total()) {
            throw std::logic_error("MaxPool2d input size does not match");
        }

        const CHWLayout inputLayout(inputShape);
        const CHWLayout outputLayout(outputShape);
        const auto oChannels = outputLayout.channels();
        const auto oWidth = outputLayout.width();
        const auto oHeight = outputLayout.height();
        for (size_t och = 0; och < oChannels; ++och) {
            for (size_t oy = 0; oy < oHeight; ++oy) {
                for (size_t ox = 0; ox < oWidth; ++ox) {
                    const size_t startY = oy * stride;
                    const size_t startX = ox * stride;
                    size_t maxIndex = inputLayout.index(och, startY, startX);
                    auto maxValue = input[maxIndex];

                    for (size_t ky = 0; ky < kernelSize; ++ky) {
                        for (size_t kx = 0; kx < kernelSize; ++kx) {
                            const size_t y = startY + ky;
                            const size_t x = startX + kx;
                            if (const size_t inputIndex = inputLayout.index(och, y, x);
                                input[inputIndex] > maxValue) {
                                maxValue = input[inputIndex];
                                maxIndex = inputIndex;
                            }
                        }
                    }

                    const size_t outputIndex = outputLayout.index(och, oy, ox);
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
        Tensor inputGradient(inputShape.total(), Scalar{0});

        for (size_t i = 0; i < outputGradient.size(); ++i) {
            inputGradient[maxIndices[i]] += outputGradient[i];
        }

        hasForwardResult = false;
        return inputGradient;
    }

    const Shape &MaxPool2d::getInputShape() const noexcept {
        return inputShape;
    }

    const Shape &MaxPool2d::getOutputShape() const noexcept {
        return outputShape;
    }

    SoftmaxCategoricalCrossEntropy::SoftmaxCategoricalCrossEntropy(const size_t classesSize)
        : probabilities(classesSize),
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
