#include "nncpp/network.hpp"

#include <cmath>
#include <stdexcept>

namespace nncpp {
    Scalar sigmoid(const Scalar value) {
        return 1.0f / (1.0f + std::exp(-value));
    }

    Scalar derivativeSigmoid(const Scalar activation) {
        return activation * (1.0f - activation);
    }

    Scalar MSE(const Scalar output, const Scalar target) {
        const Scalar difference = output - target;
        return 0.5f * difference * difference;
    }

    Tensor MSE(const Tensor &output, const Tensor &target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("output do not match output layer");
        }
        Tensor outputVector(output.size());
        for (std::size_t i = 0; i < output.size(); ++i) {
            outputVector[i] = MSE(output[i], target[i]);
        }
        return outputVector;
    }

    Tensor derivativeMSE(const Tensor &output, const Tensor &target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("output and target sizes differ");
        }

        Tensor gradient(output.size());
        for (std::size_t i = 0; i < output.size(); ++i) {
            gradient[i] = output[i] - target[i];
        }
        return gradient;
    }

    Scalar outputDelta(const Scalar output, const Scalar target) {
        return (output - target) * derivativeSigmoid(output);
    }

    void forwardPass(std::vector<std::vector<Neuron> > &network) {
        if (network.empty()) {
            throw std::invalid_argument("network must not be empty");
        }

        for (std::size_t layerIndex = 1; layerIndex < network.size(); ++layerIndex) {
            const auto &previousLayer = network[layerIndex - 1];

            for (Neuron &neuron: network[layerIndex]) {
                if (neuron.weights.size() != previousLayer.size()) {
                    throw std::logic_error("neuron weights do not match previous layer");
                }

                auto z = neuron.bias;
                for (std::size_t weightIndex = 0; weightIndex < neuron.weights.size(); ++weightIndex) {
                    z += neuron.weights[weightIndex] * previousLayer[weightIndex].activation;
                }
                neuron.activation = sigmoid(z);
            }
        }
    }

    void calculateDeltas(std::vector<std::vector<Neuron> > &network, const Tensor &targets) {
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
                Scalar dE_da = 0.0f;
                for (const Neuron &nextNeuron: nextLayer) {
                    dE_da += nextNeuron.delta * nextNeuron.weights[neuronIndex];
                }

                Neuron &neuron = network[layerIndex][neuronIndex];
                neuron.delta = dE_da * derivativeSigmoid(neuron.activation);
            }
        }
    }

    void optimizeParams(std::vector<std::vector<Neuron> > &network, const Scalar learningRate) {
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
                    const Scalar dE_dw = neuron.delta * prevLayer[weightIndex].activation;
                    neuron.weights[weightIndex] -= learningRate * dE_dw;
                }
                const Scalar dE_db = neuron.delta;
                neuron.bias -= learningRate * dE_db;
            }
        }
    }

    Scalar loss(const Tensor &output, const Tensor &targets) {
        if (output.size() != targets.size()) {
            throw std::invalid_argument("output and target sizes differ");
        }

        Scalar total = 0.0;
        for (std::size_t index = 0; index < output.size(); ++index) {
            total += MSE(output[index], targets[index]);
        }
        return total;
    }

    Scalar loss(const std::vector<Neuron> &outputs, const Tensor &targets) {
        if (outputs.size() != targets.size()) {
            throw std::invalid_argument("targets do not match output layer");
        }

        Scalar total = 0.0;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            total += MSE(outputs[index].activation, targets[index]);
        }
        return total;
    }

    Tensor &sigmoid(Tensor &input) {
        for (Scalar &index: input) {
            index = sigmoid(index);
        }
        return input;
    }
}
