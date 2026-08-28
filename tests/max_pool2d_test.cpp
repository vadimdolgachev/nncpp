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

    void expectTensor(const nncpp::Tensor &actual,
                      const nncpp::Tensor &expected,
                      const std::string &label) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error(label + " has an unexpected size");
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            expectNear(actual[i], expected[i], label);
        }
    }
}

int main() {
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::MaxPool2d(nncpp::Shape{1, 4}, 2, 2)); },
        "non-CHW input shape"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::MaxPool2d(nncpp::Shape{0, 4, 4}, 2, 2)); },
        "zero channels"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::MaxPool2d(nncpp::Shape{1, 4, 0}, 2, 2)); },
        "zero width"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::MaxPool2d(nncpp::Shape{1, 4, 4}, 0, 2)); },
        "zero kernel"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::MaxPool2d(nncpp::Shape{1, 4, 4}, 5, 2)); },
        "kernel larger than input"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::MaxPool2d(nncpp::Shape{1, 4, 4}, 2, 0)); },
        "zero stride"
    );

    nncpp::MaxPool2d pool(nncpp::Shape{2, 2, 4}, 2, 2);
    if (pool.getInputShape() != nncpp::Shape{2, 2, 4} ||
        pool.getOutputShape() != nncpp::Shape{2, 1, 2}) {
        throw std::runtime_error("output size is incorrect before forward");
    }
    expectThrows<std::logic_error>(
        [&pool] { static_cast<void>(pool.backward({1.0f, 1.0f, 1.0f, 1.0f})); },
        "backward before forward"
    );
    expectThrows<std::logic_error>(
        [&pool] { static_cast<void>(pool.forward({1.0f})); },
        "mismatched input"
    );

    const nncpp::Tensor input = {
        -4.0f, -2.0f, -8.0f, -1.0f,
        -3.0f, -5.0f, -6.0f, -7.0f,
         1.0f,  9.0f,  3.0f,  4.0f,
         8.0f,  2.0f,  7.0f,  6.0f,
    };
    expectTensor(pool.forward(input), {-2.0f, -1.0f, 9.0f, 7.0f}, "multi-channel output");
    expectThrows<std::logic_error>(
        [&pool] { static_cast<void>(pool.backward({1.0f})); },
        "mismatched output gradient"
    );
    expectTensor(
        pool.backward({1.0f, 2.0f, 3.0f, 4.0f}),
        {
            0.0f, 1.0f, 0.0f, 2.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 3.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 0.0f,
        },
        "multi-channel input gradient"
    );
    expectThrows<std::logic_error>(
        [&pool] { static_cast<void>(pool.backward({1.0f, 1.0f, 1.0f, 1.0f})); },
        "repeated backward"
    );

    nncpp::MaxPool2d overlapping(nncpp::Shape{1, 3, 3}, 2, 1);
    expectTensor(
        overlapping.forward({
            0.0f, 0.0f, 0.0f,
            0.0f, 9.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
        }),
        {9.0f, 9.0f, 9.0f, 9.0f},
        "overlapping output"
    );
    expectTensor(
        overlapping.backward({1.0f, 1.0f, 1.0f, 1.0f}),
        {
            0.0f, 0.0f, 0.0f,
            0.0f, 4.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
        },
        "overlapping input gradient"
    );

    nncpp::MaxPool2d ties(nncpp::Shape{1, 2, 4}, 2, 2);
    const nncpp::Scalar lowest = std::numeric_limits<nncpp::Scalar>::lowest();
    expectTensor(
        ties.forward(nncpp::Tensor(8, lowest)),
        {lowest, lowest},
        "lowest-value output"
    );
    expectTensor(
        ties.backward({2.0f, 3.0f}),
        {
            2.0f, 0.0f, 3.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
        },
        "first-maximum tie gradient"
    );

    return 0;
}
