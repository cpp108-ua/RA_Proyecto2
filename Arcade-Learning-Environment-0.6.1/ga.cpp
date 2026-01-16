#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <string> // Aseguramos que string esté incluido
#include "src/ale_interface.hpp"
#include "NeuralNetwork.hpp"

using namespace std;

const bool REAL_FITNESS = true;
const bool WARM_START = true;

const bool TIMESTAMPED_MODELS = true;
const string MODEL_PREFIX = "GA_";

// Parámetros del Algoritmo Genético
const int POPULATION_SIZE = 50;
const int GENERATIONS = 100;
const int FRAMES_PER_EPISODE = 3600;
const int FRAME_SKIP = 4;
const double MUTATION_RATE = 0.20; 
const double MUTATION_STRENGTH = 2.0; 
const int ELITISM_COUNT = 5;

// Arquitectura de la Red
const vector<int> TOPOLOGY = {64, 32, 18};

// Estructura para contener pares input/target del dataset
struct TrainingData {
    vector<double> inputs;
    vector<double> targets; 
};

struct Genome {
    vector<double> genes;
    double fitness; 
};

// Carga y preprocesamiento del dataset CSV
vector<TrainingData> loadData(const string& filename, int& inputSizeRef) {
    vector<TrainingData> dataset;
    ifstream file(filename);
    string line, token;

    if (!file.is_open()) return dataset;

    // --- FIX 1: Detect Column Count ---
    if(getline(file, line)) {
        stringstream ss(line);
        int cols = 0;
        while(getline(ss, token, ',')) cols++;
        inputSizeRef = cols - 1; // Last column is Action
    }

    file.clear(); 
    file.seekg(0);

    if (!isdigit(file.peek()) && file.peek() != '-') {
        getline(file, line); 
    }

    // Parsing
    while(getline(file, line)) {
        if(line.empty()) continue;
        stringstream ss(line);
        TrainingData sample;
        sample.inputs.reserve(inputSizeRef); // Optimization
        
        // Inputs
        for(int i=0; i<inputSizeRef; i++) {
            getline(ss, token, ',');
            try {
                double val = stod(token); 
                sample.inputs.push_back(val);
            } catch(...) { 
                sample.inputs.push_back(0.0); 
            }
        }
        
        // Target (Action)
        getline(ss, token, ',');
        int action = 0;
        try { action = stoi(token); } catch(...) { continue; }
        
        // Assuming TOPOLOGY is global. Ensure output layer size matches.
        int outputLayerSize = TOPOLOGY.back();
        sample.targets.resize(outputLayerSize, 0.0);
        
        if(action >= 0 && action < outputLayerSize) {
            sample.targets[action] = 1.0;
        }
        
        dataset.push_back(sample);
    }
    
    return dataset;
}
// Optimization struct (Define this above the function or in a header)
struct FastBitMap {
    int byteIdx;
    int shift;
};

double calculateRealFitness(NeuralNetwork& net, const vector<double>& genes, const vector<int>& activeGlobalBits) {
    net.setWeights(genes);

    // --- SILENCE START ---
    std::streambuf* originalCout = std::cout.rdbuf();
    std::streambuf* originalCerr = std::cerr.rdbuf();
    std::stringstream nullStream;
    std::cout.rdbuf(nullStream.rdbuf());
    std::cerr.rdbuf(nullStream.rdbuf());

    ALEInterface ale;
    ale.setBool("display_screen", false); 
    ale.setBool("sound", false);
    ale.setInt("random_seed", 123); 
    ale.setFloat("repeat_action_probability", 0.0); 
    ale.setInt("frame_skip", FRAME_SKIP);
    ale.setInt("max_num_frames_per_episode", FRAMES_PER_EPISODE);
    ale.loadROM("supported/assault.bin");

    // --- SILENCE END ---
    std::cout.rdbuf(originalCout);
    std::cerr.rdbuf(originalCerr);

    // --- PRE-CALCULATE BIT LOCATIONS (Optimization) ---
    // Do this ONCE before the game loop to save CPU
    std::vector<FastBitMap> inputMap;
    inputMap.reserve(activeGlobalBits.size());

    for (int globalID : activeGlobalBits) {
        // Convert Global ID (e.g. 87) -> Byte 10, Bit 7
        inputMap.push_back({
            globalID / 8,       // Byte Index
            7 - (globalID % 8)  // Bit Shift (MSB First)
        });
    }

    // Reuse memory for inputs
    std::vector<double> inputs;
    inputs.reserve(inputMap.size());

    double totalScore = 0.0;
    int framesWithoutPoints = 0;

    // 3. Game Loop
    while (!ale.game_over()) {
        const ALERAM& ram = ale.getRAM();
        
        // Clear vector but keep memory reserved
        inputs.clear(); 

        // FIXED: BIT-LEVEL EXTRACTION
        // Instead of dividing by 255, we extract the exact bit (0 or 1)
        for (const auto& loc : inputMap) {
            int byteVal = ram.get(loc.byteIdx);
            double bit = (byteVal >> loc.shift) & 1; // Extracts 0 or 1
            inputs.push_back(bit);
        }

        // Feed Forward
        vector<double> output = net.feedForward(inputs);

        // Select action
        auto maxIt = max_element(output.begin(), output.end());
        int actionIndex = distance(output.begin(), maxIt);
        
        double reward = ale.act(static_cast<Action>(actionIndex));
        totalScore += reward;

        // Kill Switch
        if (reward > 0) {
            framesWithoutPoints = 0; 
        } else {
            framesWithoutPoints++;
        }
        
        if (framesWithoutPoints > 400) break;
    }

    return totalScore;
}

// Evaluación del desempeño de la red
double calculateFitness(NeuralNetwork& nn, const vector<double>& genes, const vector<TrainingData>& data) {
    nn.setWeights(genes);
    double score = 0.0;
    int step = 3; // Muestreo para optimización de rendimiento

    for(size_t i = 0; i < data.size(); i += step) {
        vector<double> output = nn.feedForward(data[i].inputs);
        
        // Determinar predicción (índice con mayor activación)
        int predicted = 0;
        double maxVal = -9999.0;
        for(size_t o=0; o<output.size(); o++) {
            if(output[o] > maxVal) { maxVal = output[o]; predicted = o; }
        }

        // Determinar target real
        int actual = 0;
        for(size_t o=0; o<data[i].targets.size(); o++) {
            if(data[i].targets[o] > 0.9) { actual = o; break; }
        }

        // Sistema de Puntuación 
        if (predicted == actual) {
            if (actual == 0) score += 1.0;          // IDLE
            else if (actual == 2) score += 10.0;    // SHOOT 
            else score += 5.0;                      // MOVEMENT/OTHERS
        } 
        else {
            // Penalizaciones por errores críticos
            if (actual == 2 && predicted == 0) score -= 5.0; // No disparó
            else if (actual != 0 && predicted == 0) score -= 2.0; // Inactividad incorrecta
            else score -= 1.0; // Error genérico
        }
    }
    return score; 
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

vector<int> getActiveBytes() {
    vector<int> active;
    ifstream file("dat/bit_mask.txt"); // Reads BIT mask
    int val;
    if(file.is_open()) {
        while(file >> val) active.push_back(val);
        file.close();
    } else {
        // Fallback: Use all 1024 bits if no mask found
        for(int i=0; i<1024; i++) active.push_back(i);
        cerr << "[WARN] 'bit_mask.txt' not found. Using full 1024-bit RAM." << endl;
    }
    return active;
}
int main(int argc, char** argv) {
    // --- NUEVA LÓGICA DE INTERACCIÓN CON EL USUARIO ---
    string filename;
    vector<TrainingData> data;
    vector<int> activeGlobalBits; // For Real Fitness (Bit Mask)
    int detectedInputSize = 0;

    cout << "========================================" << endl;
    cout << "      ENTRENAMIENTO GENETICO (GA)       " << endl;
    cout << "========================================" << endl;

    // --- 1. SETUP DATA SOURCE ---
    if(!REAL_FITNESS) {
        // STATIC TRAINING (Use dataset)
        cout << "Introduce el dataset BIT-LEVEL a utilizar:" << endl;
        cout << "(Deja vacio y pulsa ENTER para usar 'dataset_bin_optimized.csv'): ";
        getline(cin, filename);
        if (filename.empty()) filename = "dataset_bin_optimized.csv";

        cout << "-> Buscando archivo: " << filename << endl;
        
        // This function now handles 0/1 bits correctly (no division by 255)
        data = loadData(filename, detectedInputSize); 

        if(data.empty()) {
            cerr << "[ERROR] No se pudo cargar el dataset." << endl;
            return -1;
        }
        cout << "Dataset cargado (" << data.size() << " muestras). Input Size: " << detectedInputSize << endl;
    } 
    else {
        // REAL FITNESS (Use Mask)
        activeGlobalBits = getActiveBytes(); // Now reads 'bit_mask.txt'
        detectedInputSize = activeGlobalBits.size();
        cout << "Modo REAL FITNESS activo." << endl;
        cout << "Input Mask cargada: " << detectedInputSize << " bits activos." << endl;
    }

    // --- 2. TOPOLOGY SETUP ---
    vector<int> topology {detectedInputSize}; // Update local variable
    topology.insert(topology.end(), TOPOLOGY.begin(), TOPOLOGY.end());

    // --- 3. WARM START ---
    string warmStartFile;
    bool useWarmStart = false;
    NeuralNetwork* referenceNN = nullptr;

    if (WARM_START) {
        cout << "\n--- WARM START (Transfer Learning) ---" << endl;
        cout << "Introduce nombre de cerebro para iniciar (Enter = 'dat/brain.txt'): ";
        getline(cin, warmStartFile);
        if (warmStartFile.empty()) warmStartFile = "dat/brain.txt";
        
        if (ifstream(warmStartFile)) {
            cout << ">>> Intentando cargar: " << warmStartFile << endl;
            try {
                // Try to load existing brain
                referenceNN = new NeuralNetwork(warmStartFile);
                
                // Validate topology match
                if(referenceNN->getInputSize() != detectedInputSize) {
                    cerr << "[ERROR] Topology Mismatch! Brain has " << referenceNN->getInputSize() 
                         << " inputs, but we need " << detectedInputSize << "." << endl;
                    cerr << "Starting from SCRATCH instead." << endl;
                    delete referenceNN;
                    referenceNN = new NeuralNetwork(topology);
                } else {
                    useWarmStart = true;
                    cout << ">>> WARM START EXITOSO." << endl;
                }
            } catch (...) {
                cerr << "[ERROR] Corrupt brain file. Starting from scratch." << endl;
                referenceNN = new NeuralNetwork(topology);
            }
        } else {
            cout << ">>> COLD START: No se encontro archivo. Iniciando desde cero." << endl;
            referenceNN = new NeuralNetwork(topology);
        }
    } else {
        referenceNN = new NeuralNetwork(topology);
    }

    NeuralNetwork dummyNN = *referenceNN; // Copy for structure
    int genomeSize = dummyNN.getWeightCount();
    
    cout << "Iniciando proceso evolutivo..." << endl;
    cout << "Poblacion: " << POPULATION_SIZE << " | Genes por individuo: " << genomeSize << endl;

    // --- 4. POPULATION INIT ---
    vector<Genome> population(POPULATION_SIZE);
    mt19937 rng(time(0));
    uniform_real_distribution<> distWeight(-1.0, 1.0);
    normal_distribution<> distMutation(0.0, 0.5);

    vector<double> baseGenes = dummyNN.getWeights();

    for(int i = 0; i < POPULATION_SIZE; i++) {
        population[i].genes.resize(genomeSize);
        population[i].fitness = -9999; 

        if (useWarmStart) {
            if (i == 0) {
                // Elitism: Copy exact best brain
                population[i].genes = baseGenes;
            } else {
                // Diversity: Mutated clones
                for(size_t k=0; k < genomeSize; k++) {
                    double gene = baseGenes[k];
                    if ((rand() % 100) < 30) gene += distMutation(rng);
                    population[i].genes[k] = gene;
                }
            }
        } 
        else {
            // Cold Start: Random Xavier-ish weights
            for(double &g : population[i].genes) g = distWeight(rng);
        }
    }
    
    // Clean up reference pointer
    delete referenceNN;

    // --- 5. EVOLUTION LOOP ---
    for(int gen = 1; gen <= GENERATIONS; ++gen) {
        
        // Parallel Evaluation
        #pragma omp parallel for
        for(int i=0; i<POPULATION_SIZE; i++) {
            // Create temp NN with correct structure
            NeuralNetwork tempNN(dummyNN.getTopology()); 
            
            if (REAL_FITNESS) 
                population[i].fitness = calculateRealFitness(tempNN, population[i].genes, activeGlobalBits);
            else
                population[i].fitness = calculateFitness(tempNN, population[i].genes, data);
        }

        // Sort by Fitness
        sort(population.begin(), population.end(), [](const Genome& a, const Genome& b) {
            return a.fitness > b.fitness;
        });

        // Logging
        if(gen % 10 == 0 || gen == 1)
            cout << "Gen " << gen << " | Best Fitness: " << population[0].fitness << endl;

        // Auto-Save Checkpoint
        if(gen % 50 == 0) {
            dummyNN.setWeights(population[0].genes);
            dummyNN.save("dat/brain_ga.txt"); 
        }

        // --- NEW POPULATION GENERATION ---
        vector<Genome> newPop;
        newPop.reserve(POPULATION_SIZE);

        // Elitism (Keep best)
        for(int i=0; i<ELITISM_COUNT; i++) newPop.push_back(population[i]);

        // Genetic Operators
        uniform_int_distribution<> distIndex(0, POPULATION_SIZE / 2); // Pick parents from top 50%
        uniform_real_distribution<> distMut(0.0, 1.0);
        normal_distribution<> distMutVal(0.0, MUTATION_STRENGTH);

        while(newPop.size() < POPULATION_SIZE) {
            const Genome& p1 = population[distIndex(rng)];
            const Genome& p2 = population[distIndex(rng)];
            Genome child;
            child.genes = p1.genes; 

            // Uniform Crossover
            for(size_t k=0; k<child.genes.size(); k++) {
                if(distMut(rng) > 0.5) child.genes[k] = p2.genes[k];
            }

            // Mutation
            for(double &g : child.genes) {
                if(distMut(rng) < MUTATION_RATE) g += distMutVal(rng);
            }
            newPop.push_back(child);
        }
        population = newPop;
    }
    
    // --- 6. FINAL SAVE ---
    dummyNN.setWeights(population[0].genes);

    stringstream timestamp = getTimeStmp();
    string outputFilename = "dat/" + MODEL_PREFIX + timestamp.str() + ".txt";

    if (TIMESTAMPED_MODELS) dummyNN.save(outputFilename);
    dummyNN.save("dat/brain.txt"); // Overwrite default for easy play

    cout << "Entrenamiento finalizado." << endl;
    return 0;
}