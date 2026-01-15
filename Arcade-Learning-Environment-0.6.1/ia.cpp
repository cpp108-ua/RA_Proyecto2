#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>
#include "src/ale_interface.hpp"
#include <SDL/SDL.h> 
#include "NeuralNetwork.hpp"

// --- FUNCIONES DE ACTIVACIÓN ---
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double relu(double x) { return (x > 0) ? x : 0; }
/*
// --- MOTOR DE INFERENCIA ---
class NeuralNetwork {
private:
    int inputNodes, hiddenNodes, outputNodes;
    std::vector<std::vector<double>> wInputHidden;
    std::vector<std::vector<double>> wHiddenOutput;
    std::vector<double> bHidden;
    std::vector<double> bOutput;
    std::vector<double> hiddenLayer; // Buffer de activaciones

public:
    NeuralNetwork(const std::vector<int>& activeBytes) {
        // Inicialización del Modelo
        if (!loadWeights("brain.txt")) {
            std::cerr << "[ERROR] No se pudo cargar 'brain.txt'." << std::endl;
            SDL_Quit();
            exit(1);
        }

        // Validación de dimensiones
        if (getInputSize() != activeBytes.size()) {
            std::cerr << "[FATAL] Dimension mismatch: Red(" << getInputSize()
                      << ") != Mascara(" << activeBytes.size() << ")." << std::endl;
            SDL_Quit();
            exit(1);
        }
    }

    // Carga de parámetros serializados
    bool loadWeights(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        // 1. Topología
        file >> inputNodes >> hiddenNodes >> outputNodes;

        // 2. Asignación de recursos
        wInputHidden.resize(inputNodes, std::vector<double>(hiddenNodes));
        wHiddenOutput.resize(hiddenNodes, std::vector<double>(outputNodes));
        bHidden.resize(hiddenNodes);
        bOutput.resize(outputNodes);
        hiddenLayer.resize(hiddenNodes);

        // 3. Deserialización de Matrices y Bias
        // Orden estricto: wIH -> bH -> wHO -> bO
        for(int i=0; i<inputNodes; i++)
            for(int h=0; h<hiddenNodes; h++)
                file >> wInputHidden[i][h];

        for(int h=0; h<hiddenNodes; h++) file >> bHidden[h];

        for(int h=0; h<hiddenNodes; h++)
            for(int o=0; o<outputNodes; o++)
                file >> wHiddenOutput[h][o];

        for(int o=0; o<outputNodes; o++) file >> bOutput[o];

        file.close();
        return true;
    }

    // Forward Propagation (Inferencia)
    int predict(const std::vector<double>& inputs) {
        
        // A. Capa Oculta (Activación ReLU)
        for(int h=0; h<hiddenNodes; h++) {
            double sum = bHidden[h];
            for(int i=0; i<inputNodes; i++) {
                sum += inputs[i] * wInputHidden[i][h];
            }
            hiddenLayer[h] = relu(sum);
        }

        // B. Capa de Salida (Activación Sigmoid + ArgMax)
        int bestAction = 0;
        double maxProb = -9999.0;

        for(int o=0; o<outputNodes; o++) {
            double sum = bOutput[o];
            for(int h=0; h<hiddenNodes; h++) {
                sum += hiddenLayer[h] * wHiddenOutput[h][o];
            }
            double outputVal = sigmoid(sum); // Probabilidad [0,1]

            // Selección Greedy
            if (outputVal > maxProb) {
                maxProb = outputVal;
                bestAction = o;
            }
        }
        return bestAction;
    }
    
    int getInputSize() const { return inputNodes; }
};
*/
// --- SISTEMA DECISIONAL ---
class DecisionSystem {
private:
    std::vector<NeuralNetwork> models; // Conjunto de modelos
    std::vector<int> actionSpace;    // Espacio de acciones
    std::vector<double> weights;       // Pesos asociados a cada modelo

public:
    
    DecisionSystem(const std::vector<int>& activeBytes, int numModels) {
        // Inicialización de múltiples modelos
        for (int i = 0; i < numModels; i++) {
            models.emplace_back(activeBytes);
            weights.push_back(1.0 / numModels); // Pesos iniciales uniformes
            actionSpace.push_back(0);
        }
    }
    
    DecisionSystem(const std::vector<std::string>& modelFiles) {
        // Inicialización desde archivos específicos
        for (const auto& file : modelFiles) {
            models.emplace_back(file);
            weights.push_back(1.0 / modelFiles.size());
            actionSpace.push_back(0);
        }
    }

    // Predicción basada en votación ponderada
    int predict(const std::vector<double>& inputs) {
        std::vector<double> actionScores(models[0].getInputSize(), 0.0);

        // Acumulación de predicciones ponderadas
        for (size_t i = 0; i < models.size(); i++) {
            int action = models[i].predict(inputs);
            actionSpace[i] = action;
            actionScores[action] += weights[i];
        }

        // Selección de la acción con mayor puntuación
        return std::distance(actionScores.begin(),
                             std::max_element(actionScores.begin(), actionScores.end()));
    }

    // Actualización de pesos (boosting)
    void updateWeightsBoosting(const std::vector<double>& errors, const std::vector<int>& gains) {
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] *= exp(-errors[i]); // Penalización basada en error
        }

        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] *= (1.0 + gains[i] * 0.1); // Recompensa por ganancia
        }

        // Normalización de pesos
        double sumWeights = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (double& weight : weights) {
            weight /= sumWeights;
        }
    }

    void updateWeightsUniform() {
        double uniformWeight = 1.0 / weights.size();
        for (double& weight : weights) {
            weight = uniformWeight;
        }
    }
};

void runInference(const std::string& romFile) {
    // Inicialización SDL/ALE
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) < 0) {
        std::cerr << "[ERROR] SDL Initialization failed." << std::endl;
        return;
    }
    ALEInterface alei;
    alei.setInt("random_seed", 123);
    alei.setBool("display_screen", true);
    alei.setBool("sound", true);
    alei.loadROM(romFile);

    // Carga de Máscara (Feature Selection)
    std::vector<int> activeBytes;
    std::ifstream maskFile("mask.txt");
    if (maskFile.is_open()) {
        int idx;
        while (maskFile >> idx) activeBytes.push_back(idx);
        maskFile.close();
        std::cout << "Mascara activa: " << activeBytes.size() << " inputs." << std::endl;
    } else {
        for (int i = 0; i < 128; i++) activeBytes.push_back(i);
        std::cout << "[WARN] Usando máscara default (128 bytes)." << std::endl;
    }

    // Inicialización del Modelo
    //DecisionSystem brain(activeBytes, 1); // Usando un solo modelo por simplicidad
    DecisionSystem brain(vector<string>({"brain.txt"}));
    std::cout << "--- INFERENCIA INICIADA (ReLU/Sigmoid) ---" << std::endl;

    // Game Loop
    while (!alei.game_over()) {
        // 1. Obtención de estado
        const ALERAM& ram = alei.getRAM();
        std::vector<double> inputs;
        inputs.reserve(activeBytes.size());

        // 2. Normalización [0, 255] -> [0.0, 1.0]
        for (int idx : activeBytes) {
            inputs.push_back(ram.get(idx) / 255.0);
        }

        // 3. Predicción
        int actionIndex = brain.predict(inputs);

        // 4. Actuación
        alei.act((Action)actionIndex);

        // SDL Events
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) exit(0);
        }
    }

    std::cout << "Sesión finalizada." << std::endl;
    SDL_Quit();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <rom_file>" << std::endl;
        return -1;
    }

    runInference(argv[1]);
    return 0;
}