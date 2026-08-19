#include "nncpp/network.hpp"
#include "test_fixture.hpp"

#include <array>
#include <stdexcept>
#include <vector>

namespace {
    double evaluateLoss(
        nncpp::DenseLayer& dense1,
        nncpp::Sigmoid& sigmoid1,
        nncpp::DenseLayer& dense2,
        nncpp::Sigmoid& sigmoid2,
        const std::vector<double>& input,
        const std::vector<double>& targets
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

    const std::vector<double> input = {0.25, 0.5};
    const std::vector<double> targets = {1.0};

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

    const std::vector<double> dense1Analytic =
        dense1.getWeightGradients();
    const std::vector<double> dense2Analytic =
        dense2.getWeightGradients();
    constexpr double epsilon = 1e-6;

    std::vector<double> dense1Weights = dense1.getWeights();
    for (std::size_t index = 0; index < dense1Weights.size(); ++index) {
        const double original = dense1Weights[index];

        dense1Weights[index] = original + epsilon;
        dense1.setWeights(dense1Weights);
        const double lossPlus =
            evaluateLoss(dense1, sigmoid1, dense2, sigmoid2, input, targets);

        dense1Weights[index] = original - epsilon;
        dense1.setWeights(dense1Weights);
        const double lossMinus =
            evaluateLoss(dense1, sigmoid1, dense2, sigmoid2, input, targets);

        dense1Weights[index] = original;
        dense1.setWeights(dense1Weights);
        const double numerical = (lossPlus - lossMinus) / (2.0 * epsilon);
        expectNear(
            dense1Analytic[index],
            numerical,
            "dense1 chain gradient",
            1e-8
        );
    }

    std::vector<double> dense2Weights = dense2.getWeights();
    for (std::size_t index = 0; index < dense2Weights.size(); ++index) {
        const double original = dense2Weights[index];

        dense2Weights[index] = original + epsilon;
        dense2.setWeights(dense2Weights);
        const double lossPlus =
            evaluateLoss(dense1, sigmoid1, dense2, sigmoid2, input, targets);

        dense2Weights[index] = original - epsilon;
        dense2.setWeights(dense2Weights);
        const double lossMinus =
            evaluateLoss(dense1, sigmoid1, dense2, sigmoid2, input, targets);

        dense2Weights[index] = original;
        dense2.setWeights(dense2Weights);
        const double numerical = (lossPlus - lossMinus) / (2.0 * epsilon);
        expectNear(
            dense2Analytic[index],
            numerical,
            "dense2 chain gradient",
            1e-8
        );
    }

    return 0;
}
