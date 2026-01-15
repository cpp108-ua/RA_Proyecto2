#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <string> // Aseguramos que string esté incluido

using namespace std;

const bool TIMESTAMPED_MODELS = true;
const string MODEL_PREFIX = "GA_";

// Parámetros del Algoritmo Genético
const int POPULATION_SIZE = 400;
const double MUTATION_RATE = 0.20; 
const double MUTATION_STRENGTH = 2.0; 
const int ELITISM_COUNT = 5;
const int GENERATIONS = 500;

// Arquitectura de la Red
const int HIDDEN_NEURONS = 64;
const int OUTPUT_NEURONS = 18;

// Estructura para contener pares input/target del dataset
struct TrainingData {
    vector<double> inputs;
    vector<double> targets; 
};

class NeuralNetwork {
private:
    int inputNodes, hiddenNodes, outputNodes;
    vector<double> weights; 

public:
    NeuralNetwork(int inputs, int hidden, int outputs) : inputNodes(inputs), hiddenNodes(hidden), outputNodes(outputs) {
        // Reserva memoria lineal para todos los pesos y bias de la red
        int totalWeights = (inputs * hidden) + hidden + (hidden * outputs) + outputs;
        weights.resize(totalWeights);
    }

    // Inicialización de pesos usando distribución uniforme
    void randomize() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.2, 0.2); 
        
        for(double &w : weights){
            w = dis(gen);
        } 
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
    vector<double> feedForward(const vector<double>& inputs) {
        int wIdx = 0;
        vector<double> hidden(hiddenNodes);
        
        // Capa Oculta (Activación ReLU)
        for(int h=0; h<hiddenNodes; h++) {
            double sum = 0;
            for(int i=0; i<inputNodes; i++) sum += inputs[i] * weights[wIdx++];
            sum += weights[wIdx++]; // Bias
            hidden[h] = relu(sum); 
        }

        // Capa de Salida (Activación Sigmoide para probabilidad)
        vector<double> outputs(outputNodes);
        for(int o=0; o<outputNodes; o++) {
            double sum = 0;
            for(int h=0; h<hiddenNodes; h++) sum += hidden[h] * weights[wIdx++];
            sum += weights[wIdx++]; // Bias
            outputs[o] = sigmoid(sum);
        }
        return outputs;
    }

    // Serialización de pesos a formato matricial para importación en motor de juego
    void saveCompatibleFormat(const string& filename) {
        ofstream file(filename);
        if(!file.is_open()) return;
        
        // Cabecera: Topología de la red
        file << inputNodes << " " << hiddenNodes << " " << outputNodes << "\n";
        
        int wIdx = 0;
        
        // Reconstrucción de matrices
        vector<vector<double>> wIH(inputNodes, vector<double>(hiddenNodes));
        vector<double> bH(hiddenNodes);
        for(int h=0; h<hiddenNodes; h++) {
            for(int i=0; i<inputNodes; i++) wIH[i][h] = weights[wIdx++];
            bH[h] = weights[wIdx++];
        }

        // Reconstrucción de matrices Hidden->Output
        vector<vector<double>> wHO(hiddenNodes, vector<double>(outputNodes));
        vector<double> bO(outputNodes);
        for(int o=0; o<outputNodes; o++) {
            for(int h=0; h<hiddenNodes; h++) wHO[h][o] = weights[wIdx++];
            bO[o] = weights[wIdx++];
        }

        // Escritura en texto plano
        for(int i=0; i<inputNodes; i++) { for(int h=0; h<hiddenNodes; h++) file << wIH[i][h] << " "; file << "\n"; }
        for(double b : bH) file << b << " "; file << "\n";
        for(int h=0; h<hiddenNodes; h++) { for(int o=0; o<outputNodes; o++) file << wHO[h][o] << " "; file << "\n"; }
        for(double b : bO) file << b << " "; file << "\n";
        file.close();
    }
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
        sample.targets.resize(OUTPUT_NEURONS, 0.0);
        if(action >= 0 && action < OUTPUT_NEURONS) sample.targets[action] = 1.0;
        
        dataset.push_back(sample);
    }
    return dataset;
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

int main(int argc, char** argv) {
    // --- NUEVA LÓGICA DE INTERACCIÓN CON EL USUARIO ---
    string filename;
    
    cout << "========================================" << endl;
    cout << "      ENTRENAMIENTO GENETICO (GA)       " << endl;
    cout << "========================================" << endl;
    cout << "Introduce el nombre del dataset a utilizar:" << endl;
    cout << "(Deja vacio y pulsa ENTER para usar 'dataset_optimized.csv'): ";

    // Capturar entrada del usuario
    getline(cin, filename);

    // Asignar valor por defecto si está vacío
    if (filename.empty()) {
        filename = "dataset_optimized.csv";
        cout << "-> Usando archivo por defecto: " << filename << endl;
    } else {
        cout << "-> Buscando archivo: " << filename << endl;
    }

    int inputSize = 0;
    vector<TrainingData> data = loadData(filename, inputSize);

    if(data.empty()) {
        cerr << "[ERROR] No se pudo cargar el dataset: '" << filename << "'" << endl;
        cerr << "Verifica que el archivo existe y tiene el formato correcto." << endl;
        return -1;
    }
    
    cout << "Dataset cargado (" << data.size() << " muestras). Input Size: " << inputSize << endl;
    cout << "Iniciando proceso evolutivo..." << endl;

    // Inicialización de Población
    NeuralNetwork dummyNN(inputSize, HIDDEN_NEURONS, OUTPUT_NEURONS);
    int genomeSize = dummyNN.getWeightCount();

    vector<Genome> population(POPULATION_SIZE);
    mt19937 rng(time(0));
    uniform_real_distribution<> distWeight(-1.0, 1.0);

    for(auto& ind : population) {
        ind.genes.resize(genomeSize);
        for(double &g : ind.genes) g = distWeight(rng);
        ind.fitness = 0;
    }

    // Bucle Principal de Evolución
    for(int gen = 1; gen <= GENERATIONS; ++gen) {
        
        // Evaluación paralela de fitness
        #pragma omp parallel for
        for(int i=0; i<POPULATION_SIZE; i++) {
            NeuralNetwork tempNN(inputSize, HIDDEN_NEURONS, OUTPUT_NEURONS);
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
            dummyNN.saveCompatibleFormat("brain_ga.txt"); 
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

    if (TIMESTAMPED_MODELS) dummyNN.saveCompatibleFormat(outputFilename);
    dummyNN.saveCompatibleFormat("brain.txt");

    cout << "Entrenamiento finalizado." << endl;
    return 0;
}