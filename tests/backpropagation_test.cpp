#include "test_fixture.hpp"

#include <array>
#include <vector>

int main() {
    constexpr double learningRate = 0.001;
    const std::vector targets = {1.0};

    nncpp::Network network = makeTestNetwork();
    nncpp::forwardPass(network);
    const double lossBefore = nncpp::loss(network.back(), targets);

    nncpp::calculateDeltas(network, targets);

    nncpp::optimizeParams(network, learningRate);

    expectNear(
        network.back().front().delta,
        -0.1093328811810161,
        "output delta"
    );

    constexpr std::array expectedHiddenDeltas = {
        -0.005445345467430105,
        0.0026990182265129265,
        -0.010729158374727777,
        -0.007082221694440901,
    };
    for (std::size_t index = 0; index < expectedHiddenDeltas.size(); ++index) {
        expectNear(
            network[1][index].delta,
            expectedHiddenDeltas[index],
            "hidden delta"
        );
    }

    const std::vector<std::vector<std::vector<double> > > expectedWeights = {
        {
            {0.10000136133636686, 0.20000272267273372},
            {-0.3000006747545566, 0.3999986504908868},
            {0.5000026822895937, -0.5999946354208127},
            {0.7000017705554236, 0.8000035411108473},
        },
        {
            {
                0.20005807865130132,
                -0.0999392093993701,
                0.4000471968199546,
                0.3000748494544912,
            },
        },
    };
    const std::vector<std::vector<double> > expectedBiases = {
        {
            5.445345467430105e-06,
            0.0999973009817735,
            -0.09998927084162527,
            0.20000708222169444,
        },
        {-0.199890667118819},
    };

    for (std::size_t layerIndex = 1; layerIndex < network.size(); ++layerIndex) {
        for (std::size_t neuronIndex = 0; neuronIndex < network[layerIndex].size(); ++neuronIndex) {
            const nncpp::Neuron &neuron = network[layerIndex][neuronIndex];

            for (std::size_t weightIndex = 0; weightIndex < neuron.weights.size(); ++weightIndex) {
                expectNear(
                    neuron.weights[weightIndex],
                    expectedWeights[layerIndex - 1][neuronIndex][weightIndex],
                    "updated weight"
                );
            }
            expectNear(
                neuron.bias,
                expectedBiases[layerIndex - 1][neuronIndex],
                "updated bias"
            );
        }
    }

    nncpp::forwardPass(network);
    const double lossAfter = nncpp::loss(network.back(), targets);
    if (lossAfter >= lossBefore) {
        throw std::runtime_error("backpropagation did not reduce loss");
    }

    return 0;
}
