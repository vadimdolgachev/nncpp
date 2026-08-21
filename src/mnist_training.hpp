#pragma once

#include "nncpp/network.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nncpp::detail::mnist {
    inline constexpr std::size_t imageSize = 28 * 28;
    inline constexpr std::size_t classCount = 10;

    template<typename Images, typename Labels>
    void validateSplit(const Images &images, const Labels &labels, const std::string &splitName) {
        if (images.empty()) {
            throw std::runtime_error(splitName + " split is empty");
        }
        if (images.size() != labels.size()) {
            throw std::runtime_error(splitName + " image and label counts differ");
        }

        for (std::size_t i = 0; i < images.size(); ++i) {
            if (images[i].size() != imageSize) {
                throw std::runtime_error(std::format("{} image {} does not contain {} pixels", splitName, i,
                                                     imageSize));
            }
            if (static_cast<std::size_t>(labels[i]) >= classCount) {
                throw std::runtime_error(std::format("{} label {} is outside [0, 9]", splitName, i));
            }
        }
    }

    template<typename Dataset>
    void validateDataset(const Dataset &dataset) {
        validateSplit(dataset.training_images, dataset.training_labels, "training");
        validateSplit(dataset.test_images, dataset.test_labels, "test");
    }

    inline std::vector<std::size_t> makeShuffledIndices(const std::size_t size, std::mt19937 &generator) {
        std::vector<std::size_t> indices(size);
        std::ranges::iota(indices, 0u);
        std::ranges::shuffle(indices, generator);
        return indices;
    }

    template<typename Image, typename Tensor>
        requires std::ranges::sized_range<Image> &&
                 std::ranges::sized_range<Tensor> &&
                 std::ranges::output_range<Tensor, Scalar> &&
                 std::convertible_to<std::ranges::range_value_t<Image>, Scalar>
    void normalizeImage(const Image &image, Tensor &input) {
        if (std::ranges::size(image) != imageSize || std::ranges::size(input) != imageSize) {
            throw std::runtime_error(
                std::format(
                    "MNIST normalization: expected size {}, got image size {} and input size {}",
                    imageSize,
                    std::ranges::size(image),
                    std::ranges::size(input)
                )
            );
        }

        std::ranges::transform(image, input.begin(), [](const auto pixel) {
            return static_cast<Scalar>(pixel) / 255;
        });
    }

    template<typename Network, typename Images, typename Labels>
    ClassificationMetrics evaluateClassification(Network &network, const Images &images, const Labels &labels) {
        validateSplit(images, labels, "evaluation");

        Tensor input(imageSize);
        Tensor target(classCount);
        SoftmaxCategoricalCrossEntropy evaluationLoss(classCount);
        double totalLoss = 0.0;
        std::size_t correct = 0;

        for (std::size_t i = 0; i < images.size(); ++i) {
            normalizeImage(images[i], input);
            std::ranges::fill(target, 0);
            const auto label = static_cast<std::size_t>(labels[i]);
            target[label] = Scalar{1};

            Tensor output = input;
            for (const auto &layer: network) {
                output = layer->forward(output);
            }

            totalLoss += evaluationLoss.forward(output, target);
            const auto predicted = static_cast<std::size_t>(
                std::distance(output.begin(), std::ranges::max_element(output))
            );
            if (predicted == label) {
                ++correct;
            }
        }

        const auto sampleCount = static_cast<double>(images.size());
        return {
            .averageLoss = totalLoss / sampleCount,
            .accuracy = static_cast<double>(correct) / sampleCount,
        };
    }
}
