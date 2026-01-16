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

// --- SISTEMA DECISIONAL ---
class DecisionSystem {
private:
    std::vector<NeuralNetwork> models; // Conjunto de modelos
    std::vector<int> actionSpace;    // Espacio de acciones
    std::vector<double> weights;       // Pesos asociados a cada modelo

public:
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