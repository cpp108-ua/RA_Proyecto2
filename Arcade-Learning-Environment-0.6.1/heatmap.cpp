#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <string> // Necesario para getline

// --- CONFIGURACIÓN ---
const int RAM_SIZE = 128;
const double NOISE_THRESHOLD = 0.0; // Subir a 0.01 si se quiere filtrar cambios mínimos

struct ByteStats {
    int id;
    double stdDev;
};

// --- UTILIDADES VISUALES ---
std::string getANSIColor(double intensity) {
    if (intensity < 0.05) return "\033[48;5;232m"; // Negro (Inactivo)
    if (intensity < 0.20) return "\033[48;5;24m";  // Azul Oscuro
    if (intensity < 0.40) return "\033[48;5;33m";  // Azul Claro
    if (intensity < 0.60) return "\033[48;5;208m"; // Naranja
    return "\033[48;5;196m";                       // Rojo (Hotspot)
}

// --- LÓGICA PRINCIPAL ---
int main(int argc, char **argv) {
    std::string filename;

    // --- INTERACCIÓN CON EL USUARIO ---
    std::cout << "========================================" << std::endl;
    std::cout << "   ANALIZADOR DE VARIANZA DE RAM (ALE)  " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Introduce el nombre del archivo CSV a analizar: " << std::endl;
    std::cout << "(Deja vacio y pulsa ENTER para usar 'dataset.csv'): ";
    
    std::getline(std::cin, filename);

    // Si el usuario no escribe nada, usamos el default
    if (filename.empty()) {
        filename = "dataset.csv";
        std::cout << "-> Usando archivo por defecto: " << filename << std::endl;
    } else {
        std::cout << "-> Buscando archivo: " << filename << std::endl;
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "\n[ERROR] No se pudo abrir el archivo '" << filename << "'" << std::endl;
        std::cerr << "Verifica que el nombre es correcto y que el archivo existe en esta carpeta." << std::endl;
        return -1;
    }

    // ACUMULADORES ESTADÍSTICOS
    std::vector<double> sum(RAM_SIZE, 0.0);
    std::vector<double> sumSq(RAM_SIZE, 0.0);
    long long n = 0;

    std::string line, token;
    
    // Detectar si el archivo tiene cabecera y saltarla
    char peek = file.peek();
    if (!isdigit(peek)) std::getline(file, line); 

    std::cout << "Analizando datos..." << std::endl;

    // PROCESAMIENTO DE DATOS
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int val;
    
        for (int i = 0; i < RAM_SIZE; ++i) {
            std::getline(ss, token, ',');
            try {
                if (!token.empty()) {
                    val = std::stoi(token);
                    sum[i] += val;
                    sumSq[i] += (val * val);
                }
            } catch (...) { continue; }
        }
        n++;
    }

    // CÁLCULO DE DESVIACIÓN ESTÁNDAR
    std::vector<ByteStats> stats(RAM_SIZE);
    double maxStdDev = 0.0;

    for (int i = 0; i < RAM_SIZE; ++i) {
        double mean = sum[i] / n;
        double variance = (sumSq[i] / n) - (mean * mean);
        if (variance < 0) variance = 0; 
        
        stats[i].id = i;
        stats[i].stdDev = std::sqrt(variance);

        if (stats[i].stdDev > maxStdDev) maxStdDev = stats[i].stdDev;
    }

    // 4. REPORTE: TOP 10 BYTES ACTIVOS
    std::vector<ByteStats> sortedStats = stats;
    std::sort(sortedStats.begin(), sortedStats.end(), [](const ByteStats& a, const ByteStats& b) {
        return a.stdDev > b.stdDev;
    });

    std::cout << "\n--- TOP 10 BYTES MÁS SIGNIFICATIVOS ---" << std::endl;
    std::cout << "Byte ID | Desviación | Interpretación Potencial" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(7) << sortedStats[i].id << " | " 
                  << std::setw(10) << std::fixed << std::setprecision(4) << sortedStats[i].stdDev << " | ";
        
        if (sortedStats[i].stdDev > 50) std::cout << "Posición / Temporizador Global";
        else if (sortedStats[i].stdDev > 10) std::cout << "Estado de Entidad / Animación";
        else std::cout << "Bandera de estado / Contador lento";
        std::cout << std::endl;
    }

    // VISUALIZACIÓN: MAPA DE CALOR
    std::cout << "\n--- MAPA DE ACTIVIDAD DE RAM (128 Bytes) ---" << std::endl;
    std::cout << "     ";
    for(int c=0; c<16; c++) std::cout << std::setw(3) << c << " ";
    std::cout << "\n    +" << std::string(16*4, '-') << "+";

    for (int row = 0; row < 8; ++row) {
        std::cout << "\n" << std::setw(3) << (row * 16) << " |";
        for (int col = 0; col < 16; ++col) {
            int index = row * 16 + col;
            double intensity = (maxStdDev > 0) ? (stats[index].stdDev / maxStdDev) : 0;
            
            std::cout << getANSIColor(intensity) 
                      << std::setw(3) << index 
                      << "\033[0m ";
        }
        std::cout << "|";
    }
    std::cout << "\nTotal Frames: " << n << std::endl;

    // EXPORTACIÓN DE MÁSCARA
    std::ofstream maskFile("mask.txt");
    int activeCount = 0;
    
    for (int i = 0; i < RAM_SIZE; ++i) {
        if (stats[i].stdDev > NOISE_THRESHOLD) { 
            maskFile << i << std::endl;
            activeCount++;
        }
    }
    maskFile.close();
    
    std::cout << "\n[OK] 'mask.txt' generado correctamente." << std::endl;
    std::cout << "Input reducido de 128 a " << activeCount << " dimensiones." << std::endl;

    return 0;
}