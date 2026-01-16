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

using namespace std;

// --- OPTIMIZATION STRUCT ---
// Maps a Neural Network input node directly to a specific bit in RAM
struct BitLocation {
    int byteIndex; // RAM Byte (0-127)
    int bitShift;  // Bit offset (0-7)
};

// --- SYSTEM DECISIONAL ---
class DecisionSystem {
private:
    std::vector<NeuralNetwork> models; // Conjunto de modelos
    std::vector<int> actionSpace;      // Espacio de acciones
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
        // FIX: Size must be 18 (ALE Max Actions), not InputSize (which could be small)
        std::vector<double> actionScores(18, 0.0);

        // Acumulación de predicciones ponderadas
        for (size_t i = 0; i < models.size(); i++) {
            int action = models[i].predict(inputs);
            
            // Safety check for action bounds
            if(action >= 0 && action < 18) {
                actionScores[action] += weights[i];
            }
        }

        // Selección de la acción con mayor puntuación
        return std::distance(actionScores.begin(),
                             std::max_element(actionScores.begin(), actionScores.end()));
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

    // --- CARGA DE MÁSCARA (BIT-LEVEL) ---
    std::vector<BitLocation> fastInputMap;
    std::ifstream maskFile("dat/bit_mask.txt"); // Prioritize the BIT mask
    
    if (maskFile.is_open()) {
        std::cout << "Loading Bit Mask (Optimized)..." << std::endl;
        int globalIdx;
        while (maskFile >> globalIdx) {
            // Convert Global ID (0-1023) back to Byte/Bit logic
            // Logic must match 'recorder_bits.cpp'
            int byteIndex = globalIdx / 8;        
            int bitOffset = 7 - (globalIdx % 8); // MSB First
            
            fastInputMap.push_back({byteIndex, bitOffset});
        }
        maskFile.close();
        std::cout << "Active Inputs: " << fastInputMap.size() << " bits." << std::endl;
    } else {
        std::cout << "[WARN] 'bit_mask.txt' not found. Defaulting to FULL RAM (1024 bits)." << std::endl;
        for(int i = 0; i < 1024; ++i) {
             fastInputMap.push_back({i / 8, 7 - (i % 8)});
        }
    }

    // Inicialización del Modelo
    DecisionSystem brain(vector<string>({"dat/BP_160126_0945.txt"}));
    
    std::cout << "--- INFERENCIA INICIADA (Bit-Level) ---" << std::endl;

    // Pre-allocate input vector to avoid re-allocation every frame
    std::vector<double> inputs;
    inputs.reserve(fastInputMap.size());

    // Game Loop
    while (!alei.game_over()) {
        // 1. Obtención de estado
        const ALERAM& ram = alei.getRAM();
        
        // Reset Inputs
        inputs.clear();

        // 2. EXTRACTION (BIT-LEVEL FAST MAP)
        for (const auto& loc : fastInputMap) {
            int byteVal = ram.get(loc.byteIndex);
            
            // Extract specific bit
            double bit = (byteVal >> loc.bitShift) & 1;
            inputs.push_back(bit);
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