#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <string>

// --- CONFIGURATION ---
const double NOISE_THRESHOLD = 0.0; // Keep 0.0 to capture any change, 0.01 to filter distinct noise

struct BitStat {
    int globalIndex; // The ID from 0 to 1023
    std::string name; // "ram_10_b7"
    double stdDev;
};

// --- HELPER: Parse "ram_10_b7" to Global Index ---
// Formula: Byte * 8 + (7 - Bit) 
// We use (7-Bit) because we push bits MSB first (7,6...0)
int parseHeaderToGlobalIndex(const std::string& header) {
    try {
        size_t ramPos = header.find("ram_");
        size_t bPos = header.find("_b");
        
        if (ramPos == std::string::npos || bPos == std::string::npos) return -1;

        int byteIdx = std::stoi(header.substr(ramPos + 4, bPos - (ramPos + 4)));
        int bitIdx = std::stoi(header.substr(bPos + 2));

        return (byteIdx * 8) + (7 - bitIdx);
    } catch (...) {
        return -1;
    }
}

int main(int argc, char **argv) {
    std::string filename;

    std::cout << "========================================" << std::endl;
    std::cout << "   BIT-LEVEL VARIANCE ANALYZER          " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Enter bit-level CSV (Enter = 'dat/dataset_bin_full.csv'): ";
    
    std::getline(std::cin, filename);
    if (filename.empty()) filename = "dat/dataset_bin_full.csv";
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open '" << filename << "'" << std::endl;
        return -1;
    }

    // --- 1. READ HEADER ---
    std::string line, token;
    std::getline(file, line); // Read first line
    std::stringstream headerSS(line);
    
    std::vector<int> columnToGlobalIndex;
    std::vector<std::string> columnNames;

    // Map columns to real RAM IDs
    while (std::getline(headerSS, token, ',')) {
        if (token == "action") break; // Stop at action
        columnNames.push_back(token);
        columnToGlobalIndex.push_back(parseHeaderToGlobalIndex(token));
    }

    int numColumns = columnNames.size();
    std::cout << "Detected " << numColumns << " bit columns." << std::endl;

    // --- 2. ACCUMULATE STATS ---
    std::vector<double> sum(numColumns, 0.0);
    std::vector<double> sumSq(numColumns, 0.0);
    long long n = 0;

    std::cout << "Analyzing variance..." << std::endl;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        double val;

        for (int i = 0; i < numColumns; ++i) {
            std::getline(ss, token, ',');
            try {
                val = std::stod(token); // Reads 0.0 or 1.0
                sum[i] += val;
                sumSq[i] += (val * val);
            } catch (...) {}
        }
        n++;
    }

    if (n == 0) {
        std::cerr << "[ERROR] No data rows found." << std::endl;
        return -1;
    }

    // --- 3. CALCULATE & EXPORT ---
    std::ofstream maskFile("dat/bit_mask.txt");
    int activeBits = 0;

    std::cout << "\n--- TOP ACTIVE BITS ---" << std::endl;
    std::cout << "Global ID | Name       | StdDev" << std::endl;

    for (int i = 0; i < numColumns; ++i) {
        double mean = sum[i] / n;
        double variance = (sumSq[i] / n) - (mean * mean);
        if (variance < 0) variance = 0;
        double stdDev = std::sqrt(variance);

        // If this bit actually changes (StdDev > 0)
        if (stdDev > NOISE_THRESHOLD) {
            // Write the GLOBAL INDEX to the mask file
            if (columnToGlobalIndex[i] != -1) {
                maskFile << columnToGlobalIndex[i] << std::endl;
                activeBits++;

                // Print top info (optional filter for screen clutter)
                if (activeBits <= 10 || stdDev > 0.4) {
                    std::cout << std::setw(9) << columnToGlobalIndex[i] << " | "
                              << std::setw(10) << columnNames[i] << " | "
                              << std::fixed << std::setprecision(4) << stdDev << std::endl;
                }
            }
        }
    }
    maskFile.close();

    std::cout << "\n[SUCCESS] Generated 'bit_mask.txt'" << std::endl;
    std::cout << "Kept " << activeBits << " active bits out of " << numColumns << "." << std::endl;

    return 0;
}