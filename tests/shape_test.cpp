#include "nncpp/network.hpp"

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
    constexpr nncpp::Shape vectorShape{7};
    static_assert(vectorShape.rank() == 1);
    static_assert(vectorShape[0] == 7);
    static_assert(vectorShape.total() == 7);

    constexpr nncpp::Shape chwShape{2, 3, 4};
    static_assert(chwShape.rank() == 3);
    static_assert(chwShape[0] == 2);
    static_assert(chwShape[1] == 3);
    static_assert(chwShape[2] == 4);
    static_assert(chwShape.total() == 24);

    constexpr nncpp::CHWLayout chwLayout(chwShape);
    static_assert(chwLayout.channels() == 2);
    static_assert(chwLayout.height() == 3);
    static_assert(chwLayout.width() == 4);
    static_assert(chwLayout.index(0, 0, 0) == 0);
    static_assert(chwLayout.index(0, 2, 3) == 11);
    static_assert(chwLayout.index(1, 0, 0) == 12);
    static_assert(chwLayout.index(1, 2, 3) == 23);
    static_assert(chwLayout.shape() == chwShape);

    constexpr nncpp::Shape emptyShape{};
    static_assert(emptyShape.rank() == 0);
    static_assert(emptyShape.total() == 0);

    constexpr nncpp::Shape zeroDimension{2, 0, 4};
    static_assert(zeroDimension.total() == 0);

    if (chwShape != nncpp::Shape{2, 3, 4} || chwShape == nncpp::Shape{2, 4, 3}) {
        throw std::runtime_error("shape equality is incorrect");
    }

    expectThrows<std::out_of_range>(
        [&chwShape] { static_cast<void>(chwShape[3]); },
        "out-of-range dimension access"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::Shape{1, 2, 3, 4, 5}); },
        "rank greater than four"
    );
    expectThrows<std::invalid_argument>(
        [] { static_cast<void>(nncpp::CHWLayout(nncpp::Shape{2, 3})); },
        "non-CHW shape"
    );
    expectThrows<std::out_of_range>(
        [&chwLayout] { static_cast<void>(chwLayout.index(2, 0, 0)); },
        "out-of-range channel"
    );
    expectThrows<std::out_of_range>(
        [&chwLayout] { static_cast<void>(chwLayout.index(0, 3, 0)); },
        "out-of-range row"
    );
    expectThrows<std::out_of_range>(
        [&chwLayout] { static_cast<void>(chwLayout.index(0, 0, 4)); },
        "out-of-range column"
    );
    expectThrows<std::overflow_error>(
        [] {
            static_cast<void>(nncpp::Shape{
                std::numeric_limits<std::size_t>::max(),
                2,
            });
        },
        "total-size overflow"
    );

    return 0;
}
