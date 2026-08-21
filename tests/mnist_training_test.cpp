#include "mnist_training.hpp"
#include "test_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using Image = std::vector<std::uint8_t>;

    struct Dataset final {
        std::vector<Image> training_images;
        std::vector<std::uint8_t> training_labels;
        std::vector<Image> test_images;
        std::vector<std::uint8_t> test_labels;
    };

    Image makeImage(const std::uint8_t value = 0) {
        return Image(nncpp::detail::mnist::imageSize, value);
    }

    Dataset makeValidDataset() {
        return {
            .training_images = {makeImage()},
            .training_labels = {0},
            .test_images = {makeImage()},
            .test_labels = {0},
        };
    }

    template<typename Callable>
    void expectRuntimeError(Callable action, const std::string &label) {
        try {
            action();
        } catch (const std::runtime_error &) {
            return;
        }
        throw std::runtime_error(label + " did not throw");
    }
}

int main() {
    nncpp::detail::mnist::validateDataset(makeValidDataset());

    auto emptyTraining = makeValidDataset();
    emptyTraining.training_images.clear();
    emptyTraining.training_labels.clear();
    expectRuntimeError(
        [&emptyTraining] { nncpp::detail::mnist::validateDataset(emptyTraining); },
        "empty training split"
    );

    auto emptyTest = makeValidDataset();
    emptyTest.test_images.clear();
    emptyTest.test_labels.clear();
    expectRuntimeError(
        [&emptyTest] { nncpp::detail::mnist::validateDataset(emptyTest); },
        "empty test split"
    );

    auto mismatchedCounts = makeValidDataset();
    mismatchedCounts.training_labels.push_back(1);
    expectRuntimeError(
        [&mismatchedCounts] { nncpp::detail::mnist::validateDataset(mismatchedCounts); },
        "mismatched image and label counts"
    );

    auto wrongImageSize = makeValidDataset();
    wrongImageSize.training_images[0].pop_back();
    expectRuntimeError(
        [&wrongImageSize] { nncpp::detail::mnist::validateDataset(wrongImageSize); },
        "wrong image size"
    );

    auto invalidLabel = makeValidDataset();
    invalidLabel.test_labels[0] = 10;
    expectRuntimeError(
        [&invalidLabel] { nncpp::detail::mnist::validateDataset(invalidLabel); },
        "invalid label"
    );

    std::mt19937 firstGenerator{42};
    std::mt19937 secondGenerator{42};
    const auto firstOrder = nncpp::detail::mnist::makeShuffledIndices(32, firstGenerator);
    const auto secondOrder = nncpp::detail::mnist::makeShuffledIndices(32, secondGenerator);
    if (firstOrder != secondOrder) {
        throw std::runtime_error("fixed shuffle seed is not deterministic");
    }

    auto sortedOrder = firstOrder;
    std::ranges::sort(sortedOrder);
    std::vector<std::size_t> expectedOrder(sortedOrder.size());
    std::iota(expectedOrder.begin(), expectedOrder.end(), std::size_t{0});
    if (sortedOrder != expectedOrder || firstOrder == expectedOrder) {
        throw std::runtime_error("shuffled indices are not a nontrivial permutation");
    }

    auto dense = std::make_unique<nncpp::DenseLayer>(
        nncpp::detail::mnist::imageSize,
        nncpp::detail::mnist::classCount
    );
    auto *denseLayer = dense.get();
    denseLayer->setWeights(nncpp::Tensor(
        nncpp::detail::mnist::imageSize * nncpp::detail::mnist::classCount,
        nncpp::Scalar{0}
    ));
    denseLayer->setBiases(nncpp::Tensor(nncpp::detail::mnist::classCount, nncpp::Scalar{0}));

    std::array<std::unique_ptr<nncpp::Layer>, 1> network = {std::move(dense)};
    const nncpp::Tensor trainingInput(nncpp::detail::mnist::imageSize, nncpp::Scalar{1});
    static_cast<void>(network.front()->forward(trainingInput));
    static_cast<void>(network.front()->backward(
        nncpp::Tensor(nncpp::detail::mnist::classCount, nncpp::Scalar{1})
    ));

    const auto weightsBefore = denseLayer->getWeights();
    const auto biasesBefore = denseLayer->getBiases();
    const auto weightGradientsBefore = denseLayer->getWeightGradients();
    const auto biasGradientsBefore = denseLayer->getBiasGradients();

    const std::vector<Image> evaluationImages = {makeImage(), makeImage(255)};
    const std::vector<std::uint8_t> evaluationLabels = {0, 0};
    const auto metrics = nncpp::detail::mnist::evaluateClassification(
        network,
        evaluationImages,
        evaluationLabels
    );

    expectNear(
        static_cast<nncpp::Scalar>(metrics.averageLoss),
        std::log(10.0),
        "evaluation loss",
        1e-6
    );
    expectNear(
        static_cast<nncpp::Scalar>(metrics.accuracy),
        1.0,
        "evaluation accuracy"
    );
    if (metrics.averageLoss < 0.0 || metrics.accuracy < 0.0 || metrics.accuracy > 1.0) {
        throw std::runtime_error("evaluation metrics are outside their expected ranges");
    }
    if (denseLayer->getWeights() != weightsBefore ||
        denseLayer->getBiases() != biasesBefore ||
        denseLayer->getWeightGradients() != weightGradientsBefore ||
        denseLayer->getBiasGradients() != biasGradientsBefore) {
        throw std::runtime_error("evaluation changed parameters or accumulated gradients");
    }

    return 0;
}
