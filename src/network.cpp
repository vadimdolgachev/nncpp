#include "nncpp/network.hpp"

#include <cmath>
#include <stdexcept>

namespace nncpp {
    double sigmoid(const double value) {
        return 1.0 / (1.0 + std::exp(-value));
    }

    double derivativeSigmoid(const double activation) {
        return activation * (1.0 - activation);
    }

    double MSE(const double output, const double target) {
        const double difference = output - target;
        return 0.5 * difference * difference;
    }

    double outputDelta(const double output, const double target) {
        return (output - target) * derivativeSigmoid(output);
    }

    void forwardPass(Network &network) {
        if (network.empty()) {
            throw std::invalid_argument("network must not be empty");
        }

        for (std::size_t layerIndex = 1; layerIndex < network.size(); ++layerIndex) {
            const auto &previousLayer = network[layerIndex - 1];

            for (Neuron &neuron: network[layerIndex]) {
                if (neuron.weights.size() != previousLayer.size()) {
                    throw std::logic_error("neuron weights do not match previous layer");
                }

                neuron.z = neuron.bias;
                for (std::size_t weightIndex = 0;
                     weightIndex < neuron.weights.size();
                     ++weightIndex) {
                    neuron.z +=
                            neuron.weights[weightIndex] *
                            previousLayer[weightIndex].activation;
                }
                neuron.activation = sigmoid(neuron.z);
            }
        }
    }

    void calculateDeltas(Network &network, const std::vector<double> &targets) {
        if (network.size() < 2) {
            throw std::invalid_argument("network must contain at least two layers");
        }
        if (targets.size() != network.back().size()) {
            throw std::invalid_argument("targets do not match output layer");
        }
        for (std::size_t neuronIndex = 0; neuronIndex < network.back().size(); ++neuronIndex) {
            Neuron &neuron = network.back()[neuronIndex];
            neuron.delta = outputDelta(neuron.activation, targets[neuronIndex]);
        }

        for (std::size_t layerIndex = network.size() - 2; layerIndex > 0; --layerIndex) {
            const auto &nextLayer = network[layerIndex + 1];
            const auto currentLayerSize = network[layerIndex].size();

            for (const Neuron &nextNeuron: nextLayer) {
                if (nextNeuron.weights.size() != currentLayerSize) {
                    throw std::logic_error("neuron weights do not match previous layer");
                }
            }

            for (std::size_t neuronIndex = 0; neuronIndex < currentLayerSize; ++neuronIndex) {
                double dE_da = 0.0;
                for (const Neuron &nextNeuron: nextLayer) {
                    dE_da += nextNeuron.delta * nextNeuron.weights[neuronIndex];
                }

                Neuron &neuron = network[layerIndex][neuronIndex];
                neuron.delta = dE_da * derivativeSigmoid(neuron.activation);
            }
        }
    }

    NetworkGradient updateGradient(const Network &network) {
        NetworkGradient gradient;

        if (network.size() < 2) {
            throw std::invalid_argument("network must contain at least two layers");
        }
        for (std::size_t layerIndex = 1; layerIndex < network.size(); ++layerIndex) {
            const auto &prevLayer = network[layerIndex - 1];


            for (size_t neuronIndex = 0; neuronIndex  < network[layerIndex].size(); ++neuronIndex) {
                if (network[layerIndex][neuronIndex].weights.size() != prevLayer.size()) {
                    throw std::logic_error("neuron weights do not match previous layer");
                }

                for (std::size_t weightIndex = 0; weightIndex < network[layerIndex][neuronIndex].weights.size(); ++weightIndex) {
                    const double dE_dw = network[layerIndex][neuronIndex].delta * prevLayer[weightIndex].activation;
                    gradient[layerIndex][neuronIndex].weights[weightIndex] += dE_dw;
                }
                const double dE_db = network[layerIndex][neuronIndex].delta;
                gradient[layerIndex][neuronIndex].bias += dE_db;
            }
        }
        return gradient;
    }

    void optimizeParams(Network &network, const double learningRate) {
        if (network.size() < 2) {
            throw std::invalid_argument("network must contain at least two layers");
        }
        if (learningRate <= 0.0) {
            throw std::invalid_argument("learning rate must be positive");
        }
        for (std::size_t layerIndex = 1; layerIndex < network.size(); ++layerIndex) {
            const auto &prevLayer = network[layerIndex - 1];

            for (Neuron &neuron: network[layerIndex]) {
                if (neuron.weights.size() != prevLayer.size()) {
                    throw std::logic_error("neuron weights do not match previous layer");
                }

                for (std::size_t weightIndex = 0; weightIndex < neuron.weights.size(); ++weightIndex) {
                    const double dE_dw = neuron.delta * prevLayer[weightIndex].activation;
                    neuron.weights[weightIndex] -= learningRate * dE_dw;
                }
                const double dE_db = neuron.delta;
                neuron.bias -= learningRate * dE_db;
            }
        }
    }

    double loss(const Layer &outputs, const std::vector<double> &targets) {
        if (outputs.size() != targets.size()) {
            throw std::invalid_argument("targets do not match output layer");
        }

        double total = 0.0;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            total += MSE(outputs[index].activation, targets[index]);
        }
        return total;
    }
}
