#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>

// CONFIGURACIÓN
const double KEEP_NOOP_CHANCE = 0.10; // Solo guardamos el 10% de los "Quiet"
const int ACTION_NOOP = 0;
const int ACTION_FIRE = 1;

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Uso: ./cleaner dataset_optimized.csv" << std::endl;
        return 1;
    }

    std::ifstream inFile(argv[1]);
    std::ofstream outFile("dataset_balanced.csv");
    std::string line, token;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int kept = 0;
    int dropped = 0;


    while(std::getline(inFile, line)) {
        if(line.empty()) continue;
        
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> seglist;
        while(std::getline(ss, segment, ',')) seglist.push_back(segment);
        
        if(seglist.empty()) continue;

        try {
            int action = std::stoi(seglist.back());

            // LOGICA DE FILTRADO
            bool keep = true;

            if (action == ACTION_NOOP) {
                if (dis(gen) > KEEP_NOOP_CHANCE) {
                    keep = false;
                }
            }

            if (keep) {
                outFile << line << "\n";
                kept++;
            } else {
                dropped++;
            }

        } catch(...) { continue; }
    }

    std::cout << "Limpieza terminada." << std::endl;
    std::cout << "Lineas guardadas: " << kept << std::endl;
    std::cout << "Lineas eliminadas (basura): " << dropped << std::endl;
    std::cout << "Archivo generado: dataset_balanced.csv" << std::endl;

    return 0;
}