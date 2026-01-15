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
const int POPULATION_SIZE = 200;
const double MUTATION_RATE = 0.20; 
const double MUTATION_STRENGTH = 2.0; 
const int ELITISM_COUNT = 5;
const int GENERATIONS = 500;

// Arquitectura de la Red
const vector<int> TOPOLOGY = {64, 32, 18};

// Estructura para contener pares input/target del dataset
struct TrainingData {
    vector<double> inputs;
    vector<double> targets; 
};

/*
class NeuralNetwork {
private:
    // Ex. {128, 64, 32, 18} -> 128 Input, 64 Hidden, 32 Hidden, 18 Output
    vector<int> topology;
    vector<double> weights; 

public:
    NeuralNetwork(const vector<int>& topologyStructure) : topology(topologyStructure) {
        int totalWeights = 0;
        
        // Calculate total weights needed for the DNA
        // For every layer i and i+1: Weights = (Node_i * Node_i+1) + Bias_i+1
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            int inputSize = topology[i];
            int outputSize = topology[i + 1];
            totalWeights += (inputSize * outputSize) + outputSize;
        }

        weights.resize(totalWeights);
    }

    // Inicialización de pesos usando distribución uniforme
    void randomize() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.2, 0.2); 
        
        for(double &w : weights) w = dis(gen); 
    }

    void setWeights(const vector<double>& newWeights) {
        if(newWeights.size() == weights.size()) weights = newWeights;
    }

    vector<double> getWeights() const { return weights; }
    int getWeightCount() const { return weights.size(); }

    // Funciones de activación
    double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    double relu(double x) { return (x > 0) ? x : 0; }

    // Forward Propagation
    vector<double> feedForward(vector<double> currentValues) {
        int wIdx = 0;

        // Loop through each connection between layers
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            
            vector<double> nextValues; // The values for the next layer
            int currentSize = topology[i];
            int nextSize = topology[i + 1];
            
            // Check if this is the very last layer (to use Sigmoid)
            bool isOutputLayer = (i == topology.size() - 2);

            // Process every neuron in the NEXT layer
            for (int n = 0; n < nextSize; ++n) {
                double sum = 0.0;
                
                // Dot Product: Previous Layer Inputs * Weights
                for (int c = 0; c < currentSize; ++c) {
                    sum += currentValues[c] * weights[wIdx++];
                }
                
                // Add Bias
                sum += weights[wIdx++];

                // Apply Activation
                if (isOutputLayer) {
                    nextValues.push_back(sigmoid(sum));
                } else {
                    nextValues.push_back(relu(sum));
                }
            }
            
            // Move forward: Next layer becomes current layer for next iteration
            currentValues = nextValues;
        }
        
        return currentValues; // Final Output
    }

    // Serialización de pesos a formato matricial para importación en motor de juego
    void saveCompatibleFormat(const string& filename) {
        ofstream file(filename);
        if(!file.is_open()) return;
        
        // 1. Header: Number of Layers
        file << topology.size() << "\n";
        
        // 2. Header: Size of each layer
        for(int size : topology) file << size << " ";
        file << "\n";
        
        // 3. Dump Weights (Formatted for readability, though flat would work too)
        int wIdx = 0;
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            int rows = topology[i];
            int cols = topology[i+1];

            // Save Matrix for this layer transition
            for(int c=0; c<cols; c++) { // For each neuron in next layer
                // Weights
                for(int r=0; r<rows; r++) file << weights[wIdx++] << " ";
                // Bias
                file << weights[wIdx++] << " "; 
                file << "\n";
            }
            file << "\n";
        }
        file.close();
    }

    int getOutputSize() const {
        return topology.back();
    }
};
*/

struct Genome {
    vector<double> genes;
    double fitness; 
};

// Carga y preprocesamiento del dataset CSV
vector<TrainingData> loadData(const string& filename, int& inputSizeRef) {
    vector<TrainingData> dataset;
    ifstream file(filename);
    string line, token;

    if (!file.is_open()) {
        return dataset; // Retorna vacio si falla
    }

    // Detectar tamaño de entrada basado en cabecera
    if(getline(file, line)) {
        stringstream ss(line);
        int cols = 0;
        while(getline(ss, token, ',')) cols++;
        inputSizeRef = cols - 1; 
    }

    // Parsing de filas
    while(getline(file, line)) {
        if(line.empty()) continue;
        stringstream ss(line);
        TrainingData sample;
        
        // Normalización de inputs
        for(int i=0; i<inputSizeRef; i++) {
            getline(ss, token, ',');
            sample.inputs.push_back(stod(token) / 255.0);
        }
        
        getline(ss, token, ',');
        int action = stoi(token);
        sample.targets.resize(TOPOLOGY.back(), 0.0);
        if(action >= 0 && action < TOPOLOGY.back()) sample.targets[action] = 1.0;
        
        dataset.push_back(sample);
    }
    return dataset;
}

double calculateRealFitness(NeuralNetwork& net, const vector<double>& genes, vector<int>& activeBytes) {
    net.setWeights(genes);

    // --- SILENCE START ---
    // Save the original buffers of cout and cerr
    std::streambuf* originalCout = std::cout.rdbuf();
    std::streambuf* originalCerr = std::cerr.rdbuf();

    // Redirect them to a "black hole" (stringstream)
    std::stringstream nullStream;
    std::cout.rdbuf(nullStream.rdbuf());
    std::cerr.rdbuf(nullStream.rdbuf());

    // 2. Setup ALE (Must be local variable for OpenMP thread safety)
    ALEInterface ale;
    ale.setBool("display_screen", false); 
    ale.setBool("sound", false);

    ale.setInt("random_seed", 123); 
    ale.setFloat("repeat_action_probability", 0.0); 
    ale.setInt("frame_skip", 4);
    ale.setInt("max_num_frames_per_episode", 7200);

    ale.loadROM("supported/assault.bin");

    // --- SILENCE END ---
    // Restore the original output so you can see your own logs (Gen 1, Best Fitness, etc.)
    std::cout.rdbuf(originalCout);
    std::cerr.rdbuf(originalCerr);

    double totalScore = 0.0;
    int framesWithoutPoints = 0;
    double previousScore = 0;

    // 3. Game Loop
    while (!ale.game_over()) {
        const ALERAM& ram = ale.getRAM();
        vector<double> inputs;
        inputs.reserve(activeBytes.size());

        // FIXED: Use index access and NORMALIZE values
        for (int idx : activeBytes) {
            inputs.push_back(ram.get(idx) / 255.0);
        }

        vector<double> output = net.feedForward(inputs);

        // Select action with highest probability
        auto maxIt = max_element(output.begin(), output.end());
        int actionIndex = distance(output.begin(), maxIt);
        
        // Cast to Action enum
        double reward = ale.act(static_cast<Action>(actionIndex));
        
        totalScore += reward;

        // OPTIMIZACIÓN: Kill Switch (Si no gana puntos en mucho tiempo)
        if (reward > 0) {
            framesWithoutPoints = 0; // ¡Ganó puntos! Reiniciamos contador
        } else {
            framesWithoutPoints++;
        }
        
        // Si pasan 400 'actuaciones' (aprox 25-30 segs de juego real) sin puntos, fuera.
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
    // Carga de Máscara (Feature Selection)
    std::vector<int> activeBytes;
    std::ifstream maskFile("mask.txt");
    if (maskFile.is_open()) {
        int idx;
        while (maskFile >> idx) activeBytes.push_back(idx);
        maskFile.close();
    } else {
        for (int i = 0; i < 128; i++) activeBytes.push_back(i);
        std::cout << "[WARN] Usando máscara default (128 bytes)." << std::endl;
    }
    return activeBytes;
}

int main(int argc, char** argv) {
    // --- NUEVA LÓGICA DE INTERACCIÓN CON EL USUARIO ---
    string filename;
    vector<TrainingData> data;
    
    cout << "========================================" << endl;
    cout << "      ENTRENAMIENTO GENETICO (GA)       " << endl;
    cout << "========================================" << endl;
    if(!REAL_FITNESS) {
        cout << "Introduce el nombre del dataset a utilizar:" << endl;
        cout << "(Deja vacio y pulsa ENTER para usar 'dataset_optimized.csv'): ";
        // Capturar entrada del usuario
        getline(cin, filename);
        // Asignar valor por defecto si está vacío
        if (filename.empty()) {
            filename = "dataset_optimized.csv";
            cout << "-> Usando archivo por defecto: " << filename << endl;
        } else cout << "-> Buscando archivo: " << filename << endl;

        int inputSize = 0;
        data = loadData(filename, inputSize);

        if(data.empty()) {
            cerr << "[ERROR] No se pudo cargar el dataset: '" << filename << "'" << endl;
            cerr << "Verifica que el archivo existe y tiene el formato correcto." << endl;
            return -1;}
        cout << "Dataset cargado (" << data.size() << " muestras). Input Size: " << inputSize << endl;
    }
    string warmStartFile;
    bool useWarmStart = false;
    // Inicialización de Población
    
    if (WARM_START) {
        cout << "\n--- WARM START (Transfer Learning) ---" << endl;
        cout << "Introduce nombre de cerebro para iniciar (Enter = 'brain.txt'): ";
        getline(cin, warmStartFile);
        if (warmStartFile.empty()) warmStartFile = "brain.txt";
            if (ifstream(warmStartFile)) {
            useWarmStart = true;
            cout << ">>> WARM START ACTIVO: Iniciando poblacion basada en " << warmStartFile << endl;
        } else {
            cout << ">>> COLD START: No se encontro " << warmStartFile << ". Iniciando desde cero." << endl;
        }
    } 
    NeuralNetwork dummyNN(WARM_START ? NeuralNetwork(warmStartFile) : NeuralNetwork(TOPOLOGY));

    int genomeSize = dummyNN.getWeightCount();
    cout << "Iniciando proceso evolutivo..." << endl;

    vector<Genome> population(POPULATION_SIZE);
    mt19937 rng(time(0));
    uniform_real_distribution<> distWeight(-1.0, 1.0);
    normal_distribution<> distMutation(0.0, 0.5);

    vector<double> baseGenes = dummyNN.getWeights();

    for(int i = 0; i < POPULATION_SIZE; i++) {
        population[i].genes.resize(genomeSize);
        population[i].fitness = -9999; // Reset fitness

        if (useWarmStart) {
            // ELITISM: El individuo 0 es una COPIA EXACTA
            if (i == 0) {
                population[i].genes = baseGenes;
            } 
            // DIVERSITY: El resto son clones con mutaciones (para explorar mejoras)
            else {
                for(size_t k=0; k < genomeSize; k++) {
                    double gene = baseGenes[k];
                    // Aplicar mutación suave al gen base
                    if ((rand() % 100) < 30) { // 30% chance de mutar cada peso
                        gene += distMutation(rng);
                    }
                    population[i].genes[k] = gene;
                }
            }
        } 
        else {
            // COLD START: Ruido aleatorio total
            for(double &g : population[i].genes) g = distWeight(rng);
        }
    }

    vector<int> activeBytes;
    if (REAL_FITNESS) {
        activeBytes = getActiveBytes();
        cout << "Mascara cargada: " << activeBytes.size() << " inputs." << endl;
    }

    // Bucle Principal de Evolución
    for(int gen = 1; gen <= GENERATIONS; ++gen) {
        
        // Evaluación paralela de fitness
        #pragma omp parallel for
        for(int i=0; i<POPULATION_SIZE; i++) {
            NeuralNetwork tempNN(dummyNN.getTopology());
            if (REAL_FITNESS) 
                population[i].fitness = calculateRealFitness(tempNN, population[i].genes, activeBytes);
            else
                population[i].fitness = calculateFitness(tempNN, population[i].genes, data);
        }

        // Selección (Ordenamiento descendente por fitness)
        sort(population.begin(), population.end(), [](const Genome& a, const Genome& b) {
            return a.fitness > b.fitness;
        });

        if(gen % 10 == 0 || gen == 1)
            cout << "Gen " << gen << " | Best Fitness: " << population[0].fitness << endl;

        if(gen % 50 == 0) {
            dummyNN.setWeights(population[0].genes);
            dummyNN.save("brain_ga.txt"); 
        }

        // Generación de nueva población
        vector<Genome> newPop;
        newPop.reserve(POPULATION_SIZE);

        // Elitismo
        for(int i=0; i<ELITISM_COUNT; i++) newPop.push_back(population[i]);

        // Distribuciones para operadores genéticos
        uniform_int_distribution<> distIndex(0, POPULATION_SIZE / 2); // Selección de padres
        uniform_real_distribution<> distMut(0.0, 1.0);
        normal_distribution<> distMutVal(0.0, MUTATION_STRENGTH);

        // Crossover y Mutación
        while(newPop.size() < POPULATION_SIZE) {
            const Genome& p1 = population[distIndex(rng)];
            const Genome& p2 = population[distIndex(rng)];
            Genome child;
            child.genes = p1.genes; 

            // Crossover Uniforme
            for(size_t k=0; k<child.genes.size(); k++) {
                if(distMut(rng) > 0.5) child.genes[k] = p2.genes[k];
            }

            // Mutación
            for(double &g : child.genes) {
                if(distMut(rng) < MUTATION_RATE) g += distMutVal(rng);
            }
            newPop.push_back(child);
        }
        population = newPop;
    }
    
    // Guardado final del mejor modelo
    dummyNN.setWeights(population[0].genes);

    stringstream timestamp = getTimeStmp();

    string outputFilename = MODEL_PREFIX + timestamp.str() + ".txt";

    if (TIMESTAMPED_MODELS) dummyNN.save(outputFilename);
    dummyNN.save("brain.txt");

    cout << "Entrenamiento finalizado." << endl;
    return 0;
}