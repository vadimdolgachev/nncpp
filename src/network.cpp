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

    std::vector<double> MSE(const std::vector<double> &output, const std::vector<double> &target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("output do not match output layer");
        }
        std::vector<double> outputVector(output.size());
        for (std::size_t i = 0; i < output.size(); ++i) {
            outputVector[i] = MSE(output[i], target[i]);
        }
        return outputVector;
    }

    std::vector<double> derivativeMSE(const std::vector<double> &output, const std::vector<double> &target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("output and target sizes differ");
        }

        std::vector<double> gradient(output.size());
        for (std::size_t i = 0; i < output.size(); ++i) {
            gradient[i] = output[i] - target[i];
        }
        return gradient;
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

                double z = neuron.bias;
                for (std::size_t weightIndex = 0; weightIndex < neuron.weights.size(); ++weightIndex) {
                    z += neuron.weights[weightIndex] * previousLayer[weightIndex].activation;
                }
                neuron.activation = sigmoid(z);
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

    double loss(const std::vector<double> &output, const std::vector<double> &targets) {
        double total = 0.0;
        for (std::size_t index = 0; index < output.size(); ++index) {
            total += MSE(output[index], targets[index]);
        }
        return total;
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

    std::vector<double> &sigmoid(std::vector<double> &input) {
        for (double &index: input) {
            index = sigmoid(index);
        }
        return input;
    }
}
