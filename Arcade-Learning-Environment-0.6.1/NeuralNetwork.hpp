#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <string>
#include <stdexcept>

using namespace std;

// --- ACTIVATION FUNCTIONS ---
enum ActivationType { RELU, SIGMOID, TANH, LEAKY_RELU };

class NeuralNetwork {
private:
    // ARCHITECTURE
    vector<int> topology;      // e.g. {128, 64, 32, 18}
    vector<double> weights;    // Flattened weights for GA compatibility
    
    // TRAINING STATE (Cached for Backprop)
    vector<vector<double>> layers; // Activations for each layer: layers[0]=Input, layers[L]=Output
    vector<vector<double>> sums;   // Pre-activation sums (z) for derivatives

    // HYPERPARAMETERS
    ActivationType hiddenActivation;
    ActivationType outputActivation;
    double learningRate = 0.01;
    double weightDecay = 0.00001;

    // --- HELPER: ACTIVATION MATH ---
    double derivativeFromActivation(double a, ActivationType type) {
        switch(type) {
            case SIGMOID: return a * (1.0 - a);
            case TANH:    return 1.0 - a * a;
            default:      return 1.0; // unused
        }
    }

    double derivativeFromSum(double z, ActivationType type) {
        switch(type) {
            case RELU: return (z > 0) ? 1.0 : 0.0;
            case LEAKY_RELU: return (z > 0) ? 1.0 : 0.01;
            default: return 1.0; // unused
        }
    }

    double activate(double z, ActivationType type) {
        switch(type) {
            case RELU:       return (z > 0) ? z : 0.0;
            case LEAKY_RELU: return (z > 0) ? z : 0.01 * z;
            case SIGMOID:    return 1.0 / (1.0 + exp(-z));
            case TANH:       return tanh(z);
            default:         return z;
        }
    }

public:
    // --- CONSTRUCTOR ---
    NeuralNetwork(const vector<int>& topologyStructure, 
                  ActivationType hidden = RELU, 
                  ActivationType output = SIGMOID) 
        : topology(topologyStructure), hiddenActivation(hidden), outputActivation(output) {
        int totalWeights = 0;
        // Calculate total weights needed (Matrices + Biases flattened)
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            // Weights (In * Out) + Biases (Out)
            totalWeights += (topology[i] * topology[i+1]) + topology[i+1];
        }
        weights.resize(totalWeights);
        
        // Resize cache structures for Backprop
        layers.resize(topology.size());
        sums.resize(topology.size());
        for(size_t i=0; i<topology.size(); i++) {
            layers[i].resize(topology[i]);
            sums[i].resize(topology[i]);
        }

        initializeWeights();
    }

    // FROM FILE CONSTRUCTOR
    NeuralNetwork(std::string filename) {
        if(!load(filename)) {
            throw std::runtime_error("Failed to load brain: " + filename);
        }
    }

    // --- INITIALIZATION ---
    void initializeWeights() {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        int idx = 0;
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            int fanIn  = topology[i];
            int fanOut = topology[i + 1];

            double scale = sqrt(2.0 / fanIn);

            for (int n = 0; n < fanOut; ++n) {
                for (int c = 0; c < fanIn; ++c)
                    weights[idx++] = dist(gen) * scale;

                weights[idx++] = 0.0; 
            }
        }
    }

    void sampleWeights(int startIdx, int endIdx) const {
        cout << "--- SAMPLE WEIGHTS FROM INDEX " << startIdx << " TO " << endIdx << " ---" << endl;
        for (int i = startIdx; i < endIdx; i++)
            cout << weights[i] << endl;
    }

    void setHyperparameters(double lr, double decay) {
        learningRate = lr;
        weightDecay = decay;
    }

    // --- FORWARD PASS (Works for N Layers) ---
    vector<double> feedForward(const vector<double>& inputs) {
        if(inputs.size() != topology[0]) 
            throw invalid_argument("Input size mismatch");

        layers[0] = inputs; // Layer 0 is Input

        int wIdx = 0;
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            int currentSize = topology[i];
            int nextSize = topology[i+1];
            
            bool isOutput = (i == topology.size() - 2);
            ActivationType act = isOutput ? outputActivation : hiddenActivation;

            // Compute Next Layer
            for (int n = 0; n < nextSize; ++n) {
                double sum = 0.0;
                
                // 1. Weights * Inputs
                for (int c = 0; c < currentSize; ++c) {
                    sum += layers[i][c] * weights[wIdx++];
                }
                
                // 2. Add Bias
                sum += weights[wIdx++]; // The bias is stored after the weights for this neuron

                sums[i+1][n] = sum; // Cache 'z'
                layers[i+1][n] = activate(sum, act);
            }
        }
        return layers.back();
    }

    // --- BACKPROPAGATION (Training) ---
    void train(const vector<double>& inputs, const vector<double>& targets) {
        feedForward(inputs);

        // --- Output layer deltas ---
        int L = topology.size() - 1;
        vector<double> deltas = layers[L];

        for (int i = 0; i < topology[L]; ++i) {
            double error = targets[i] - layers[L][i];
            if (outputActivation == SIGMOID || outputActivation == TANH)
                deltas[i] = error * derivativeFromActivation(layers[L][i], outputActivation);
            else
                deltas[i] = error * derivativeFromSum(sums[L][i], outputActivation);
        }

        // --- Backprop through layers ---
        for (int layer = L - 1; layer >= 0; --layer) {
            int currentSize = topology[layer];
            int nextSize    = topology[layer + 1];

            // Compute weight block start
            int layerWeightStart = 0;
            for (int k = 0; k < layer; ++k)
                layerWeightStart += (topology[k] + 1) * topology[k + 1];

            vector<double> newDeltas(currentSize, 0.0);

            for (int n = 0; n < nextSize; ++n) {
                int neuronWeightStart = layerWeightStart + n * (currentSize + 1);

                // --- Bias update ---
                int biasIdx = neuronWeightStart + currentSize;
                weights[biasIdx] += learningRate * deltas[n];

                // --- Weights update + delta propagation ---
                for (int c = 0; c < currentSize; ++c) {
                    int wIdx = neuronWeightStart + c;

                    newDeltas[c] += weights[wIdx] * deltas[n];

                    double grad = deltas[n] * layers[layer][c];
                    weights[wIdx] += learningRate * (grad - weightDecay * weights[wIdx]);
                }
            }

            // --- Apply activation derivative (skip input layer) ---
            if (layer > 0) {
                for (int c = 0; c < currentSize; ++c)
                    if (hiddenActivation == SIGMOID || hiddenActivation == TANH)
                        newDeltas[c] *= derivativeFromActivation(layers[layer][c], hiddenActivation);
                    else
                        newDeltas[c] *= derivativeFromSum(sums[layer][c], hiddenActivation);
            }

            deltas = newDeltas;
        }
    }

    // --- GENETIC ALGORITHM HELPERS ---
    void setWeights(const vector<double>& newWeights) {
        if(newWeights.size() == weights.size()) weights = newWeights;
        else cerr << "[Error] DNA size mismatch!" << endl;
    }
    
    vector<double> getWeights() const { return weights; }
    
    // --- SERIALIZATION (Unified Format) ---
    void save(const string& filename) {
        ofstream file(filename);
        if(!file.is_open()) return;

        // Header: Topology size + Layer sizes
        file << topology.size() << "\n";
        for(int t : topology) file << t << " ";
        file << "\n";

        // Header: Activation Functions
        file << static_cast<int>(hiddenActivation) << " " 
             << static_cast<int>(outputActivation) << "\n";

        // Body: All weights flat
        file.precision(10);
        for(double w : weights) file << w << " ";
        file << "\n";
        file.close();
    }

    bool load(const string& filename) {
        ifstream file(filename);
        if(!file.is_open()) return false;

        int numLayers;
        if(!(file >> numLayers)) return false;

        topology.resize(numLayers);
        int totalNodes = 0;
        for(int i=0; i<numLayers; i++) file >> topology[i];

        int hiddenAct, outputAct;
        if(!(file >> hiddenAct >> outputAct)) return false;
        hiddenActivation = static_cast<ActivationType>(hiddenAct);
        outputActivation = static_cast<ActivationType>(outputAct);

        // Re-allocate memory based on loaded topology
        int totalWeights = 0;
        for (size_t i = 0; i < topology.size() - 1; ++i) 
            totalWeights += (topology[i] * topology[i+1]) + topology[i+1];
        
        weights.resize(totalWeights);
        layers.resize(numLayers); 
        sums.resize(numLayers);
        for(int i=0; i<numLayers; i++) {
            layers[i].resize(topology[i]);
            sums[i].resize(topology[i]);
        }

        for(int i=0; i<totalWeights; i++) file >> weights[i];
        return true;
    }

    // Forward Propagation (Inferencia)
    int predict(const std::vector<double>& inputs) {
        vector<double> outputs = feedForward(inputs);
        return max_element(outputs.begin(), outputs.end()) - outputs.begin();
    }

    int getWeightCount() const { 
        return weights.size(); 
    }

    int getInputSize() const {
        return topology[0];
    }

    vector<int> getTopology() const {
        return topology;
    }
};

#endif