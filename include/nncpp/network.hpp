#pragma once

#include <vector>

namespace nncpp {
    struct Neuron final {
        std::vector<double> weights;
        double bias = 0.0;
        double activation = 0.0;
        double z = 0.0;
        double delta = 0.0;
    };
    using Layer = std::vector<Neuron>;
    using Network = std::vector<Layer>;

    struct NeuronGradient final {
        std::vector<double> weights;
        double bias = 0.0;
    };
    using LayerGradient = std::vector<NeuronGradient>;
    using NetworkGradient = std::vector<LayerGradient>;

    double sigmoid(double value);

    double derivativeSigmoid(double activation);

    double MSE(double output, double target);

    double outputDelta(double output, double target);

    void forwardPass(Network &network);

    void calculateDeltas(Network &network, const std::vector<double> &targets);

    NetworkGradient updateGradient(const Network &network);

    void optimizeParams(Network &network, double learningRate);

    double loss(const Layer &outputs, const std::vector<double> &targets);
}
