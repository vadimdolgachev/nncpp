#include "nncpp/network.hpp"
#include "mnist_training.hpp"
#include "mnist/mnist_reader.hpp"

#include <array>
#include <chrono>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace {
    constexpr nncpp::Scalar learningRate = 0.01f;
    constexpr nncpp::Scalar errorEpsilon = 1e-4f;
    constexpr std::size_t maxTrainIterations = 1'000'000;

    std::vector<std::vector<nncpp::Neuron> > makeNetwork(const nncpp::Tensor &input) {
        std::vector<std::vector<nncpp::Neuron> > network(3);
        network[0].resize(2);
        network[1].resize(4);
        network[2].resize(1);

        if (input.size() != network.front().size()) {
            throw std::invalid_argument("input does not match input layer");
        }
        for (std::size_t index = 0; index < input.size(); ++index) {
            network.front()[index].activation = input[index];
        }

        network[1][0].weights = {0.1f, 0.2f};
        network[1][1].weights = {-0.3f, 0.4f};
        network[1][1].bias = 0.1f;
        network[1][2].weights = {0.5f, -0.6f};
        network[1][2].bias = -0.1f;
        network[1][3].weights = {0.7f, 0.8f};
        network[1][3].bias = 0.2f;

        network[2][0].weights = {0.2f, -0.1f, 0.4f, 0.3f};
        network[2][0].bias = -0.2f;

        return network;
    }

    void simpleNetworkExample() {
        const nncpp::Tensor input = {0.25, 0.5};
        const nncpp::Tensor targets = {1.0};

        std::vector<std::vector<nncpp::Neuron> > network = makeNetwork(input);

        std::size_t iterations = 0;
        for (; iterations < maxTrainIterations; ++iterations) {
            nncpp::forwardPass(network);
            if (nncpp::loss(network.back(), targets) < errorEpsilon) {
                break;
            }
            nncpp::calculateDeltas(network, targets);
            nncpp::optimizeParams(network, learningRate);
        }

        nncpp::forwardPass(network);
        const nncpp::Scalar finalError = nncpp::loss(network.back(), targets);

        std::cout << "iterations: " << iterations << '\n';
        std::cout << "output: " << network.back().front().activation << '\n';
        std::cout << "error: " << finalError << '\n';

        if (finalError >= errorEpsilon) {
            throw std::runtime_error("training did not converge");
        }
    }

    void newNetworkExample() {
        const nncpp::Tensor input = {0.25, 0.5};
        const nncpp::Tensor targets = {1.0};

        std::array<std::unique_ptr<nncpp::Layer>, 4> network = {
            std::make_unique<nncpp::DenseLayer>(2, 4),
            std::make_unique<nncpp::Sigmoid>(4),
            std::make_unique<nncpp::DenseLayer>(4, 1),
            std::make_unique<nncpp::Sigmoid>(1),
        };

        size_t iterations = 0;
        for (; iterations < maxTrainIterations; ++iterations) {
            nncpp::Tensor out = input;
            for (const auto &layer: network) {
                out = layer->forward(out);
            }

            if (nncpp::loss(out, targets) < errorEpsilon) {
                break;
            }

            for (const auto &layer: network) {
                layer->resetGradients();
            }

            auto gradient = nncpp::derivativeMSE(out, targets);

            for (const auto &layer: std::views::reverse(network)) {
                gradient = layer->backward(gradient);
            }

            for (const auto &layer: network) {
                layer->applyGradient(learningRate, 1);
            }
        }

        nncpp::Tensor out = input;
        for (const auto &layer: network) {
            out = layer->forward(out);
        }
        const auto finalError = nncpp::loss(out, targets);
        std::cout << "iterations: " << iterations << '\n';
        std::cout << "output: " << out[0] << '\n';
        std::cout << "error: " << finalError << '\n';

        if (finalError >= errorEpsilon) {
            throw std::runtime_error("training did not converge");
        }
    }

    void MNISTTrainingExample() {
        constexpr size_t maxEpochs = 50;
        constexpr size_t batchSize = 32;
        constexpr std::uint32_t shuffleSeed = 42;

        const auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
        nncpp::detail::mnist::validateDataset(dataset);

        constexpr size_t hiddenLayerSize = 128;
        std::array<std::unique_ptr<nncpp::Layer>, 3> network = {
            std::make_unique<nncpp::DenseLayer>(nncpp::detail::mnist::imageSize, hiddenLayerSize),
            std::make_unique<nncpp::ReLU>(hiddenLayerSize),
            std::make_unique<nncpp::DenseLayer>(hiddenLayerSize, nncpp::detail::mnist::classCount)
        };

        nncpp::SoftmaxCategoricalCrossEntropy loss(nncpp::detail::mnist::classCount);

        const size_t trainingSize = dataset.training_images.size();
        nncpp::Tensor input(nncpp::detail::mnist::imageSize);
        nncpp::Tensor target(nncpp::detail::mnist::classCount);
        std::mt19937 batchShuffleGenerator{shuffleSeed};

        auto startTime = std::chrono::high_resolution_clock::now();

        for (size_t epoch = 0; epoch < maxEpochs; ++epoch) {
            auto batchStartTime = std::chrono::high_resolution_clock::now();
            double epochLoss = 0.0;
            const auto trainingIndices = nncpp::detail::mnist::makeShuffledIndices(trainingSize, batchShuffleGenerator);

            for (size_t batchBegin = 0; batchBegin < trainingSize; batchBegin += batchSize) {
                const size_t actualBatchSize = std::min(batchSize, trainingSize - batchBegin);

                // Start accumulating gradients for a new batch.
                for (const auto &layer: network) {
                    layer->resetGradients();
                }

                for (size_t n = 0; n < actualBatchSize; ++n) {
                    const auto index = trainingIndices[batchBegin + n];
                    nncpp::detail::mnist::normalizeImage(dataset.training_images[index], input);

                    const size_t label = dataset.training_labels[index];
                    std::ranges::fill(target, nncpp::Scalar{0});
                    target[label] = nncpp::Scalar{1};

                    // Forward
                    nncpp::Tensor out = input;
                    for (const auto &layer: network) {
                        out = layer->forward(out);
                    }

                    // Loss
                    epochLoss += loss.forward(out, target);

                    // dE / d(logits)
                    auto gradient = loss.backward();

                    // Backward
                    for (const auto &layer: std::views::reverse(network)) {
                        gradient = layer->backward(gradient);
                    }
                }

                for (const auto &layer: network) {
                    layer->applyGradient(learningRate, actualBatchSize);
                }
            }

            epochLoss /= static_cast<double>(trainingSize);
            const auto [averageLoss, accuracy] =
                nncpp::detail::mnist::evaluateClassification(network, dataset.test_images, dataset.test_labels);
            const auto batchSpentTime = std::chrono::high_resolution_clock::now() - batchStartTime;
            std::cout << std::format(
                "epoch: {}/{}, spent: {}, train loss: {:.5f}, test loss: {:.5f}, test accuracy: {:.5f}%",
                epoch,
                maxEpochs,
                std::chrono::duration_cast<std::chrono::milliseconds>(batchSpentTime),
                epochLoss,
                averageLoss,
                accuracy * 100
            ) << '\n';
        }
        const auto spentTime = std::chrono::high_resolution_clock::now() - startTime;
        std::cout << std::format("Total time spent: {}", std::chrono::duration_cast<std::chrono::seconds>(spentTime));
    }
}

int main() {
    simpleNetworkExample();
    newNetworkExample();
    MNISTTrainingExample();
    return 0;
}
