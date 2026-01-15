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
    double weightDecay = 0.001;

    // --- HELPER: ACTIVATION MATH ---
    double activate(double x, ActivationType type) {
        switch(type) {
            case SIGMOID: return 1.0 / (1.0 + exp(-x));
            case RELU: return (x > 0) ? x : 0.0;
            case TANH: return tanh(x);
            case LEAKY_RELU: return (x > 0) ? x : 0.01 * x;
            default: return 0.0;
        }
    }

    double derivative(double x, ActivationType type) {
        switch(type) {
            case SIGMOID: return x * (1.0 - x); 
            case TANH: return 1.0 - x * x;
            case RELU: return (x > 0) ? 1.0 : 0.0;
            case LEAKY_RELU: return (x > 0) ? 1.0 : 0.01;
            default: return 0.0;
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
    NeuralNetwork(string filename) {
        ifstream file(filename);
        if(!file.is_open()) throw runtime_error("Cannot open file: " + filename);

        // Read Topology
        int layersCount;
        file >> layersCount;
        topology.resize(layersCount);
        for(int i=0; i<layersCount; i++) file >> topology[i];

        // Calculate total weights
        int totalWeights = 0;
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            totalWeights += (topology[i] * topology[i+1]) + topology[i+1];
        }
        weights.resize(totalWeights);

        // Read Weights
        for(double &w : weights) file >> w;
        file.close();

        // Resize cache structures for Backprop
        layers.resize(topology.size());
        sums.resize(topology.size());
        for(size_t i=0; i<topology.size(); i++) {
            layers[i].resize(topology[i]);
            sums[i].resize(topology[i]);
        }

        // Default Activations
        hiddenActivation = RELU;
        outputActivation = SIGMOID;
    }

    // --- INITIALIZATION ---
    void initializeWeights() {
        random_device rd;
        mt19937 gen(rd());
        // Xavier/He Initialization approximation
        uniform_real_distribution<> dis(-0.5, 0.5); 
        for(double &w : weights) w = dis(gen);
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
                layers[i+1][n] = activate(sum, act); // Cache 'a'
            }
        }
        return layers.back();
    }

    // --- BACKPROPAGATION (Training) ---
    void train(const vector<double>& inputs, const vector<double>& targets) {
        // 1. Forward Pass to fill 'layers' and 'sums'
        feedForward(inputs);

        // 2. Output Errors
        vector<double> errors = layers.back();
        vector<double> nextLayerDeltas;
        
        // Calculate Output Deltas
        for(size_t i=0; i<errors.size(); i++) {
            double error = targets[i] - layers.back()[i];
            // Gradient = Error * Derivative
            nextLayerDeltas.push_back(error * derivative(layers.back()[i], outputActivation));
        }

        // 3. Backpropagate through layers (Reverse loop)
        // We start from the weights connecting LastHidden -> Output
        int wIdx = weights.size(); 

        for (int i = topology.size() - 2; i >= 0; --i) {
            int currentSize = topology[i];
            int nextSize = topology[i+1];

            // Calculate Start Index for this layer 'i'
            int layerWeightStart = 0;
            for(int k=0; k<i; k++) 
                layerWeightStart += (topology[k] * topology[k+1]) + topology[k+1];
            
            // Current Layer Deltas (to be calculated for next iteration)
            vector<double> currentLayerDeltas(currentSize, 0.0);

            // Update Weights & Calc Deltas
            // Structure in memory: [Neuron 0 weights + bias], [Neuron 1 weights + bias]...
            for (int n = 0; n < nextSize; ++n) {
                double delta = nextLayerDeltas[n];
                
                // The bias is at the end of this neuron's block
                int biasIdx = layerWeightStart + (n * (currentSize + 1)) + currentSize;
                
                // Update Bias
                weights[biasIdx] += learningRate * delta;

                // Update Weights connecting prev layer to this neuron
                for (int c = 0; c < currentSize; ++c) {
                    int weightIdx = layerWeightStart + (n * (currentSize + 1)) + c;
                    
                    // Backpropagate Error to current layer (accumulate)
                    currentLayerDeltas[c] += weights[weightIdx] * delta;
                    
                    // Update Weight
                    double decay = weightDecay * weights[weightIdx];
                    weights[weightIdx] += learningRate * delta * layers[i][c] - decay;
                }
            }

            // Prepare Deltas for the next step up (apply derivative)
            ActivationType prevAct = (i == 0) ? RELU : hiddenActivation; 
            if(i > 0) {
                 for(int c=0; c<currentSize; c++) {
                     currentLayerDeltas[c] *= derivative(layers[i][c], hiddenActivation);
                 }
            }
            nextLayerDeltas = currentLayerDeltas;
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

        // Body: All weights flat
        // (Saving flat is safer/easier than reconstructing matrices for text)
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