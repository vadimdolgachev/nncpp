#include "nncpp/network.hpp"
#include "test_fixture.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace {
    template<typename Exception, typename Callable>
    void expectThrows(Callable action, const std::string &label) {
        try {
            action();
        } catch (const Exception &) {
            return;
        }
        throw std::runtime_error(label + " did not throw");
    }
}

int main() {
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::ReLU(nncpp::Shape{0})); },
        "zero-sized ReLU"
    );

    nncpp::ReLU relu(nncpp::Shape{3});
    expectThrows<std::logic_error>(
        [&relu] { static_cast<void>(relu.backward({1.0f, 1.0f, 1.0f})); },
        "ReLU backward before forward"
    );
    expectThrows<std::logic_error>(
        [&relu] { static_cast<void>(relu.forward({1.0f, 2.0f})); },
        "ReLU mismatched input"
    );

    const nncpp::Tensor reluOutput = relu.forward({-2.0f, 0.0f, 3.0f});
    expectNear(reluOutput[0], 0.0, "ReLU negative output");
    expectNear(reluOutput[1], 0.0, "ReLU zero output");
    expectNear(reluOutput[2], 3.0, "ReLU positive output");

    expectThrows<std::logic_error>(
        [&relu] { static_cast<void>(relu.backward({1.0f, 2.0f})); },
        "ReLU mismatched gradient"
    );
    const nncpp::Tensor reluGradient = relu.backward({1.0f, 2.0f, 3.0f});
    expectNear(reluGradient[0], 0.0, "ReLU negative gradient");
    expectNear(reluGradient[1], 0.0, "ReLU zero gradient");
    expectNear(reluGradient[2], 3.0, "ReLU positive gradient");
    expectThrows<std::logic_error>(
        [&relu] { static_cast<void>(relu.backward({1.0f, 1.0f, 1.0f})); },
        "repeated ReLU backward"
    );

    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::SoftmaxCategoricalCrossEntropy(0)); },
        "zero-class softmax loss"
    );

    nncpp::SoftmaxCategoricalCrossEntropy loss(3);
    expectThrows<std::logic_error>(
        [&loss] { static_cast<void>(loss.backward()); },
        "softmax backward before forward"
    );
    expectThrows<std::invalid_argument>(
        [&loss] { static_cast<void>(loss.forward({1.0f, 2.0f}, {0.0f, 1.0f, 0.0f})); },
        "softmax mismatched logits"
    );
    expectThrows<std::invalid_argument>(
        [&loss] { static_cast<void>(loss.forward({1.0f, 2.0f, 3.0f}, {1.0f})); },
        "softmax mismatched target"
    );
    expectThrows<std::invalid_argument>(
        [&loss] { static_cast<void>(loss.forward({1.0f, 2.0f, 3.0f}, {1.0f, -0.1f, 0.1f})); },
        "softmax negative target"
    );
    expectThrows<std::invalid_argument>(
        [&loss] { static_cast<void>(loss.forward({1.0f, 2.0f, 3.0f}, {0.2f, 0.2f, 0.2f})); },
        "softmax unnormalized target"
    );
    expectThrows<std::invalid_argument>(
        [&loss] {
            static_cast<void>(loss.forward(
                {1.0f, std::numeric_limits<nncpp::Scalar>::infinity(), 3.0f},
                {0.0f, 1.0f, 0.0f}
            ));
        },
        "softmax non-finite logits"
    );

    const nncpp::Scalar ordinaryLoss =
        loss.forward({1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 1.0f});
    expectNear(ordinaryLoss, 0.4076059644, "softmax ordinary loss", 1e-6);
    const nncpp::Tensor ordinaryGradient = loss.backward();
    expectNear(ordinaryGradient[0], 0.0900305732, "softmax gradient 0", 1e-6);
    expectNear(ordinaryGradient[1], 0.2447284711, "softmax gradient 1", 1e-6);
    expectNear(ordinaryGradient[2], -0.3347590443, "softmax gradient 2", 1e-6);
    expectThrows<std::logic_error>(
        [&loss] { static_cast<void>(loss.backward()); },
        "repeated softmax backward"
    );

    const nncpp::Scalar extremeLoss =
        loss.forward({1000.0f, 0.0f, -1000.0f}, {0.0f, 1.0f, 0.0f});
    expectNear(extremeLoss, 1000.0, "softmax extreme loss");
    const nncpp::Tensor extremeGradient = loss.backward();
    expectNear(extremeGradient[0], 1.0, "softmax extreme gradient 0");
    expectNear(extremeGradient[1], -1.0, "softmax extreme gradient 1");
    expectNear(extremeGradient[2], 0.0, "softmax extreme gradient 2");


    const nncpp::Scalar equalHugeLoss =
        loss.forward({1e30f, 1e30f, -1e30f}, {1.0f, 0.0f, 0.0f});
    expectNear(equalHugeLoss, std::log(2.0), "softmax equal huge logits", 1e-6);
    nncpp::Tensor logits = {0.2f, -0.1f, 0.7f};
    const nncpp::Tensor target = {0.2f, 0.3f, 0.5f};
    static_cast<void>(loss.forward(logits, target));
    const nncpp::Tensor analyticGradient = loss.backward();
    constexpr nncpp::Scalar epsilon = 1e-3f;

    for (std::size_t index = 0; index < logits.size(); ++index) {
        const nncpp::Scalar original = logits[index];

        logits[index] = original + epsilon;
        const nncpp::Scalar lossPlus = loss.forward(logits, target);

        logits[index] = original - epsilon;
        const nncpp::Scalar lossMinus = loss.forward(logits, target);

        logits[index] = original;
        const nncpp::Scalar numericalGradient =
            (lossPlus - lossMinus) / (2.0f * epsilon);
        expectNear(
            analyticGradient[index],
            numericalGradient,
            "softmax finite-difference gradient",
            3e-4
        );
    }

    return 0;
}
