#include "test_fixture.hpp"

#include <array>

int main() {
    nncpp::Network network = makeTestNetwork();
    nncpp::forwardPass(network);

    constexpr std::array expectedHiddenActivations = {
        0.5312093733737563,
        0.5560138905446199,
        0.4316800165217519,
        0.6846015003234307,
    };

    for (std::size_t index = 0;
         index < expectedHiddenActivations.size();
         ++index) {
        expectNear(
            network[1][index].activation,
            expectedHiddenActivations[index],
            "hidden activation"
        );
    }

    expectNear(
        network.back().front().activation,
        0.5569253497362925,
        "output activation"
    );
    return 0;
}
