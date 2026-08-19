#include "nncpp/network.hpp"
#include "test_fixture.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace {
    nncpp::Scalar evaluateLoss(
        nncpp::DenseLayer& dense1,
        nncpp::Sigmoid& sigmoid1,
        nncpp::DenseLayer& dense2,
        nncpp::Sigmoid& sigmoid2,
        const nncpp::Tensor& input,
        const nncpp::Tensor& targets
    ) {
        const auto z1 = dense1.forward(input);
        const auto a1 = sigmoid1.forward(z1);
        const auto z2 = dense2.forward(a1);
        const auto output = sigmoid2.forward(z2);
        return nncpp::MSE(output, targets).front();
    }
}

int main() {
    nncpp::Sigmoid standaloneSigmoid(3);

    bool rejectedMissingForward = false;
    try {
        static_cast<void>(standaloneSigmoid.backward({1.0, 1.0, 1.0}));
    } catch (const std::logic_error&) {
        rejectedMissingForward = true;
    }
    if (!rejectedMissingForward) {
        throw std::runtime_error("sigmoid backward accepted a missing forward pass");
    }

    const auto sigmoidOutput = standaloneSigmoid.forward({0.0, 1.0, -1.0});
    constexpr std::array<double, 3> expectedSigmoid = {
        0.5,
        0.7310585786300049,
        0.2689414213699951,
    };
    for (std::size_t index = 0; index < expectedSigmoid.size(); ++index) {
        expectNear(
            sigmoidOutput[index],
            expectedSigmoid[index],
            "sigmoid output"
        );
    }

    const auto mseGradient =
        nncpp::derivativeMSE({0.2, 0.8}, {0.0, 1.0});
    expectNear(mseGradient[0], 0.2, "MSE gradient 0");
    expectNear(mseGradient[1], -0.2, "MSE gradient 1");

    const nncpp::Tensor input = {0.25f, 0.5f};
    const nncpp::Tensor targets = {1.0f};
    bool rejectedMismatchedLoss = false;
    try {
        static_cast<void>(nncpp::loss({0.2, 0.8}, {1.0}));
    } catch (const std::invalid_argument&) {
        rejectedMismatchedLoss = true;
    }
    if (!rejectedMismatchedLoss) {
        throw std::runtime_error("loss accepted mismatched target size");
    }


    nncpp::DenseLayer dense1(2, 2);
    dense1.setWeights({0.1, 0.2, -0.3, 0.4});
    dense1.setBiases({0.05, -0.1});
    nncpp::Sigmoid sigmoid1(2);

    nncpp::DenseLayer dense2(2, 1);
    dense2.setWeights({0.7, -0.5});
    dense2.setBiases({0.2});
    nncpp::Sigmoid sigmoid2(1);

    const auto z1 = dense1.forward(input);
    const auto a1 = sigmoid1.forward(z1);
    const auto z2 = dense2.forward(a1);
    const auto output = sigmoid2.forward(z2);

    auto gradient = nncpp::derivativeMSE(output, targets);
    gradient = sigmoid2.backward(gradient);
    gradient = dense2.backward(gradient);
    gradient = sigmoid1.backward(gradient);
    static_cast<void>(dense1.backward(gradient));

    const nncpp::Tensor dense1Analytic =
        dense1.getWeightGradients();
    const nncpp::Tensor dense2Analytic =
        dense2.getWeightGradients();
    constexpr nncpp::Scalar epsilon = 1e-3f;

    nncpp::Tensor dense1Weights = dense1.getWeights();
    for (std::size_t index = 0; index < dense1Weights.size(); ++index) {
        const nncpp::Scalar original = dense1Weights[index];

        dense1Weights[index] = original + epsilon;
        dense1.setWeights(dense1Weights);
        const nncpp::Scalar lossPlus =
            evaluateLoss(dense1, sigmoid1, dense2, sigmoid2, input, targets);

        dense1Weights[index] = original - epsilon;
        dense1.setWeights(dense1Weights);
        const nncpp::Scalar lossMinus =
            evaluateLoss(dense1, sigmoid1, dense2, sigmoid2, input, targets);

        dense1Weights[index] = original;
        dense1.setWeights(dense1Weights);
        const nncpp::Scalar numerical = (lossPlus - lossMinus) / (2.0f * epsilon);
        expectNear(
            dense1Analytic[index],
            numerical,
            "dense1 chain gradient",
            2e-4
        );
    }

    nncpp::Tensor dense2Weights = dense2.getWeights();
    for (std::size_t index = 0; index < dense2Weights.size(); ++index) {
        const nncpp::Scalar original = dense2Weights[index];

        dense2Weights[index] = original + epsilon;
        dense2.setWeights(dense2Weights);
        const nncpp::Scalar lossPlus =
            evaluateLoss(dense1, sigmoid1, dense2, sigmoid2, input, targets);

        dense2Weights[index] = original - epsilon;
        dense2.setWeights(dense2Weights);
        const nncpp::Scalar lossMinus =
            evaluateLoss(dense1, sigmoid1, dense2, sigmoid2, input, targets);

        dense2Weights[index] = original;
        dense2.setWeights(dense2Weights);
        const nncpp::Scalar numerical = (lossPlus - lossMinus) / (2.0f * epsilon);
        expectNear(
            dense2Analytic[index],
            numerical,
            "dense2 chain gradient",
            2e-4
        );
    }

    return 0;
}
