#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <map>
#include <iomanip>
#include "NeuralNetwork.hpp"

using namespace std;

const string MODEL_PREFIX = "BP_";
const bool TIMESTAMPED_MODELS = true;

// --- HIPERPARÁMETROS ---
const double LEARNING_RATE = 0.001;
const double WEIGHT_DECAY = 0.0001;
const double VALIDATION_SPLIT = 0.2; 
const int EARLY_STOPPING_PATIENCE = 50;
const int EPOCHS = 1000;            
const vector<int> TOPOLOGY = {64, 32, 18};

// --- FUNCIONES DE ACTIVACIÓN ---
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoidDerivative(double x) { return x * (1.0 - x); }

// ReLU: Excelente para capas ocultas
double relu(double x) { return (x > 0) ? x : 0; }
double reluDerivative(double x) { return (x > 0) ? 1.0 : 0.0; }

// Leaky ReLU 
double leakyRelu(double x) { return (x > 0) ? x : 0.01 * x; }
double leakyReluDerivative(double x) { return (x > 0) ? 1.0 : 0.01; }

// GELU
double gelu(double x) { return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3)))); }
double geluDerivative(double x) { 
    double tanhPart = tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3)));
    double sech2 = 1.0 - tanhPart * tanhPart;
    return 0.5 * (1.0 + tanhPart) + (0.5 * x * sech2 * sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x));
}

// Swish
double swish(double x) { return x * sigmoid(x); }
double swishDerivative(double x) { 
    double sig = sigmoid(x);
    return sig + x * sig * (1.0 - sig);
}

// ELU
double elu(double x) { return (x >= 0) ? x : (exp(x) - 1); }
double eluDerivative(double x) { return (x >= 0) ? 1.0 : exp(x); }

struct TrainingData {
    vector<double> inputs;
    vector<double> targets;
    int actionID;
};
/*
// --- RED NEURONAL ---
class NeuralNetwork {
private:
    int inputNodes, hiddenNodes, outputNodes;
    vector<vector<double>> wInputHidden;
    vector<vector<double>> wHiddenOutput;
    vector<double> bHidden;
    vector<double> bOutput;
    vector<double> hiddenLayer;
    vector<double> outputLayer;
    string hiddenActivationFunc;
    string outputActivationFunc;

public:
    NeuralNetwork(vector<int> topology, string hiddenActivation="relu", string outputActivation="sigmoid") 
        : inputNodes(topology[0]), hiddenNodes(topology[1]), outputNodes(topology[2]) {
        hiddenLayer.resize(hiddenNodes);
        outputLayer.resize(outputNodes);
        bHidden.resize(hiddenNodes);
        bOutput.resize(outputNodes);
        wInputHidden.resize(inputNodes, vector<double>(hiddenNodes));
        wHiddenOutput.resize(hiddenNodes, vector<double>(outputNodes));
        hiddenActivationFunc = hiddenActivation;
        outputActivationFunc = outputActivation;
        initializeWeights();
    }

    void initializeWeights() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.5, 0.5);

        for(auto& row : wInputHidden) for(auto& w : row) w = dis(gen);
        for(auto& row : wHiddenOutput) for(auto& w : row) w = dis(gen);
        for(auto& b : bHidden) b = dis(gen);
        for(auto& b : bOutput) b = dis(gen);
    }

    vector<double> feedForward(const vector<double>& inputs) {
        // Input -> Hidden (ReLU)
        for(int h=0; h<hiddenNodes; h++) {
            double sum = bHidden[h];
            for(int i=0; i<inputNodes; i++) sum += inputs[i] * wInputHidden[i][h];
            hiddenLayer[h] = getActivationFunction(hiddenActivationFunc)(sum);
        }
        
        // Hidden -> Output (Sigmoid)
        for(int o=0; o<outputNodes; o++) {
            double sum = bOutput[o];
            for(int h=0; h<hiddenNodes; h++) sum += hiddenLayer[h] * wHiddenOutput[h][o];
            outputLayer[o] = getActivationFunction(outputActivationFunc)(sum);
        }
        return outputLayer;
    }

    double (*getActivationFunction(const string& activationType))(double) {
        if (activationType == "relu") {
            return relu;
        } else if (activationType == "sigmoid") {
            return sigmoid;
        } else if (activationType == "leaky_relu") {
            return leakyRelu;
        } else if (activationType == "gelu") {
            return gelu;
        } else if (activationType == "swish") {
            return swish;
        } else if (activationType == "elu") {
            return elu;
        } else {
            throw invalid_argument("Unsupported activation function: " + activationType);
        }
    }

    double (*getActivationDerivative(const string& activationType))(double) {
        if (activationType == "relu") {
            return reluDerivative;
        } else if (activationType == "sigmoid") {
            return sigmoidDerivative;
        } else if (activationType == "leaky_relu") {
            return leakyReluDerivative;
        } else if (activationType == "gelu") {
            return geluDerivative;
        } else if (activationType == "swish") {
            return swishDerivative;
        } else if (activationType == "elu") {
            return eluDerivative;
        } else {
            throw invalid_argument("Unsupported activation function derivative: " + activationType);
        }
    }

    void train(const vector<double>& inputs, const vector<double>& targets) {
        feedForward(inputs);

        // Error Output
        vector<double> outputErrors(outputNodes), outputGradients(outputNodes);
        for(int o=0; o<outputNodes; o++) {
            double error = targets[o] - outputLayer[o];
            outputErrors[o] = error;
            outputGradients[o] = error * getActivationDerivative(outputActivationFunc)(outputLayer[o]);
        }

        // Error Hidden
        vector<double> hiddenErrors(hiddenNodes), hiddenGradients(hiddenNodes);
        for(int h=0; h<hiddenNodes; h++) {
            double error = 0.0;
            for(int o=0; o<outputNodes; o++) error += outputErrors[o] * wHiddenOutput[h][o];
            hiddenErrors[h] = error;
            hiddenGradients[h] = error * getActivationDerivative(hiddenActivationFunc)(hiddenLayer[h]);
        }

        // Update Weights (Hidden -> Output)
        for(int h=0; h<hiddenNodes; h++) {
            for(int o=0; o<outputNodes; o++) {
                double decay = WEIGHT_DECAY * wHiddenOutput[h][o];
                wHiddenOutput[h][o] += LEARNING_RATE * outputGradients[o] * hiddenLayer[h] - decay;
            }
        }
        // Update Biases (Output)
        for(int o=0; o<outputNodes; o++) bOutput[o] += LEARNING_RATE * outputGradients[o];

        // Update Weights (Input -> Hidden)
        for(int i=0; i<inputNodes; i++) {
            for(int h=0; h<hiddenNodes; h++) {
                double decay = WEIGHT_DECAY * wInputHidden[i][h];
                wInputHidden[i][h] += LEARNING_RATE * hiddenGradients[h] * inputs[i] - decay;
            }
        }
        // Update Biases (Hidden)
        for(int h=0; h<hiddenNodes; h++) bHidden[h] += LEARNING_RATE * hiddenGradients[h];
    }

    void save(const string& filename) {
        ofstream file(filename);
        if (!file.is_open()) return;
        file << inputNodes << " " << hiddenNodes << " " << outputNodes << "\n";
        for(auto& row : wInputHidden) { for(auto& w : row) file << w << " "; file << "\n"; }
        for(auto& b : bHidden) file << b << " "; file << "\n";
        for(auto& row : wHiddenOutput) { for(auto& w : row) file << w << " "; file << "\n"; }
        for(auto& b : bOutput) file << b << " "; file << "\n";
        file.close();
    }
};
*/

// --- CARGA Y BALANCEO DE DATOS ---
vector<TrainingData> loadSmartBalancedData(const string& filename, int& inputSizeRef) {
    vector<TrainingData> groupNoop; 
    vector<TrainingData> groupMove; 
    vector<TrainingData> groupFire; 
    
    ifstream file(filename);
    string line, token;

    if (!file.is_open()) return {};

    // Detectar tamaño de inputs
    if(getline(file, line)) {
        stringstream ss(line);
        int cols = 0;
        while(getline(ss, token, ',')) cols++;
        inputSizeRef = cols - 1; 
    }
    
    // Reiniciar lectura
    file.clear();
    file.seekg(0);
    
    // Saltar cabecera si existe
    char peek = file.peek();
    if (!isdigit(peek)) getline(file, line); 

    cout << "Clasificando datos..." << endl;

    while(getline(file, line)) {
        if(line.empty()) continue;
        stringstream ss(line);
        TrainingData sample;
        
        for(int i=0; i<inputSizeRef; i++) {
            getline(ss, token, ',');
            try {
                sample.inputs.push_back(stod(token) / 255.0);
            } catch(...) { sample.inputs.push_back(0); }
        }

        getline(ss, token, ',');
        int action = 0;
        try { action = stoi(token); } catch(...) { continue; }
        
        sample.actionID = action;
        sample.targets.resize(TOPOLOGY.back(), 0.0);
        if(action >= 0 && action < TOPOLOGY.back()) sample.targets[action] = 1.0;

        // --- LÓGICA DE CLASIFICACIÓN ---
        if (action == 0) {
            groupNoop.push_back(sample);
        } 
        else if (action == 2 || action == 10 || action == 11 || action == 1) { 
            groupFire.push_back(sample);
        } 
        else {
            groupMove.push_back(sample);
        }
    }
    
    size_t maxActiveSize = 0;
    if (!groupMove.empty()) maxActiveSize = max(maxActiveSize, groupMove.size());
    if (!groupFire.empty()) maxActiveSize = max(maxActiveSize, groupFire.size());

    size_t targetSize = max(maxActiveSize, size_t(500));
    

    vector<TrainingData> finalDataset;
    random_device rd;
    mt19937 g(rd());

    // Función lambda para rellenar (Upsampling circular)
    auto fillGroup = [&](vector<TrainingData>& source, int count) {
        if (source.empty()) return;
        for(int i=0; i<count; i++) {
            finalDataset.push_back(source[i % source.size()]); 
        }
    };

    shuffle(groupNoop.begin(), groupNoop.end(), g);
    shuffle(groupMove.begin(), groupMove.end(), g);
    shuffle(groupFire.begin(), groupFire.end(), g);

    // Rellenamos basándonos en el tamaño dinámico calculado
    fillGroup(groupNoop, targetSize / 2); // Menos NoOp para priorizar acción
    fillGroup(groupMove, targetSize);
    fillGroup(groupFire, targetSize); 

    shuffle(finalDataset.begin(), finalDataset.end(), g);
    
    cout << "Dataset final preparado con " << finalDataset.size() << " muestras." << endl;
    return finalDataset;
}

// Obtener la fecha y hora actual para el nombre del archivo
stringstream getTimeStmp() {
    time_t now = time(0);
    tm* localTime = localtime(&now);
    stringstream timestamp;
    timestamp << setw(2) << setfill('0') << localTime->tm_mday
              << setw(2) << setfill('0') << (1 + localTime->tm_mon)
              << setw(2) << setfill('0') << (localTime->tm_year % 100) 
              << "_"
              << setw(2) << setfill('0') << localTime->tm_hour
              << setw(2) << setfill('0') << localTime->tm_min;

    return timestamp;
}

int main(int argc, char** argv) {
    string filename;

    // --- INTERACCIÓN CON EL USUARIO ---
    cout << "========================================" << endl;
    cout << "      ENTRENADOR DE RED NEURONAL        " << endl;
    cout << "========================================" << endl;
    cout << "Introduce el dataset a usar (Enter = 'dataset_optimized.csv'): ";
    
    getline(cin, filename);
    if (filename.empty()) filename = "dataset_optimized.csv";

    cout << "-> Cargando: " << filename << endl;

    int inputSize = 0;
    vector<TrainingData> data = loadSmartBalancedData(filename, inputSize);

    if (data.empty()) {
        cerr << "[ERROR] No se pudieron cargar datos o el archivo esta vacio." << endl;
        return -1;
    }

    stringstream timestamp = getTimeStmp();

    string outputFilename = MODEL_PREFIX + timestamp.str() + ".txt";

    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);

    // Dividir datos en entrenamiento y validación
    size_t validationSize = static_cast<size_t>(data.size() * VALIDATION_SPLIT);
    vector<TrainingData> validationData(data.end() - validationSize, data.end());
    data.resize(data.size() - validationSize);

    cout << "Datos divididos: " << endl;
    cout << " - Entrenamiento: " << data.size() << " muestras" << endl;
    cout << " - Validación: " << validationData.size() << " muestras" << endl;

    vector<int> topology = {inputSize};
    topology.insert(topology.end(), TOPOLOGY.begin(), TOPOLOGY.end());
    NeuralNetwork nn(topology);
    
    cout << "Entrenando (1000 Epochs)..." << endl;

    // VARIABLES FOR EARLY STOPPING
    double bestValidationError = 1e9;
    int patience = EARLY_STOPPING_PATIENCE;     // How many epochs before stopping
    int patienceCounter = 0;                    // Counter for bad epochs
    int bestEpoch = 0;                
    
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {

        // Training Phase
        shuffle(data.begin(), data.end(), g);
        double trainError = 0.0;
        
        for (const auto& sample : data) {
            nn.train(sample.inputs, sample.targets);
            
            // Calculo simple de error para visualización
            vector<double> output = nn.feedForward(sample.inputs);
            for(size_t i=0; i<output.size(); i++) {
                double diff = sample.targets[i] - output[i];
                trainError += diff * diff;
            }
        }
        trainError /= data.size();

        // Validation Phase
        double valError = 0.0;
        for (const auto& sample : validationData) {
            vector<double> output = nn.feedForward(sample.inputs);
            for(size_t i=0; i<output.size(); i++) {
                double diff = sample.targets[i] - output[i];
                valError += diff * diff;
            }
        }
        valError /= validationData.size();
        
        // Logging
        if (epoch % 10 == 0 || epoch == 1) {
            cout << "Epoch " << setw(4) << epoch 
                 << " | Train Err: " << fixed << setprecision(5) << trainError 
                 << " | Val Err: " << valError;
        }

        // Early Stopping Check
        if (valError < bestValidationError) {
            // Found a better model
            bestValidationError = valError;
            bestEpoch = epoch;
            patienceCounter = 0; // Reset patience
            
            // Save this specific model because it's the best so far
            if (TIMESTAMPED_MODELS) nn.save(outputFilename); 
            nn.save("brain.txt");
            if (epoch % 10 == 0) cout << " [NEW BEST SAVED]";
        } else {
            // No improvement
            patienceCounter++;
            if (epoch % 10 == 0) cout << " [No improv: " << patienceCounter << "]";
        } if (epoch % 10 == 0 || epoch == 1) cout << endl;

        // Trigger Stop
        if (patienceCounter >= patience) {
            cout << "\nEARLY STOPPING TRIGGERED!" << endl;
            cout << "Stopping at epoch " << epoch << " because validation error hasn't improved in " << patience << " epochs." << endl;
            cout << "Best model was at Epoch " << bestEpoch << " with Error: " << bestValidationError << endl;
            break;
        }
    }

    cout << "\n[DONE] Entrenamiento finalizado." << endl;
    cout << "Pesos se guardarán en: " << outputFilename << endl;

    return 0;
}