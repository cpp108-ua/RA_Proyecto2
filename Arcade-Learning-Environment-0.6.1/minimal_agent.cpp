#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include "src/ale_interface.hpp"
#include <SDL/SDL.h> 

using namespace std;

// Estructura para mapear un Índice Global (0-1023) a su ubicación real en RAM
struct BitLocation {
    int globalIndex; // ID único (ej: 87)
    int byteIndex;   // Byte de RAM (ej: 10)
    int bitOffset;   // Bit dentro del byte (ej: 7, 6..0)
    
    // Constructor que convierte Global -> Local
    BitLocation(int globalIdx) : globalIndex(globalIdx) {
        byteIndex = globalIdx / 8;        // 87 / 8 = 10
        bitOffset = 7 - (globalIdx % 8);  // 87 % 8 = 7 -> Bit 0 (MSB logic)
    }
};

void usage(char const* pname) {
    cerr << "\nUSAGE:\n" << "    " << pname << " <romfile>\n";
    exit(-1);
}

int main(int argc, char **argv) {
    if (argc < 2) usage(argv[0]);

    // --- 1. CONFIGURACIÓN DE MÁSCARA ---
    vector<BitLocation> activeBits;
    string outputFilename;
    
    ifstream maskFile("dat/bit_mask.txt"); // Buscamos la máscara de BITS específica
    bool hasBitMask = maskFile.is_open(); 

    if (hasBitMask) {
        // MODO OPTIMIZADO: Usar solo los bits indicados en bit_mask.txt
        int gIdx;
        while (maskFile >> gIdx) {
            if (gIdx >= 0 && gIdx < 1024) {
                activeBits.emplace_back(gIdx);
            }
        }
        maskFile.close();
        outputFilename = "dat/dataset_bin_optimized.csv";
        
        cout << "Modo Optimizado: Máscara detectada." << endl;
        cout << "Grabando solo " << activeBits.size() << " bits activos." << endl;
        
    } else {
        // MODO COMPLETO: Grabar todos los 1024 bits (128 bytes * 8)
        // Ignoramos mask.txt (de bytes) explícitamente como se pidió.
        for(int i = 0; i < 1024; ++i) {
            activeBits.emplace_back(i);
        }
        outputFilename = "dat/dataset_bin_full.csv";
        
        cout << "Modo Completo: No se encontró 'bit_mask.txt'." << endl;
        cout << "Grabando la RAM completa (1024 bits)." << endl;
    }

    cout << "Archivo de salida: " << outputFilename << endl;

    // --- 2. INICIALIZACIÓN DE ENTORNO ---
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) < 0) return -1;

    ALEInterface alei{};
    alei.setInt("random_seed", 123);
    alei.setBool("display_screen", true); 
    alei.setBool("sound", true);
    alei.loadROM(argv[1]);
    
    // --- 3. PREPARACIÓN DE DATASET ---
    bool fileExists = false;
    ifstream fCheck(outputFilename);
    if (fCheck.good() && fCheck.peek() != ifstream::traits_type::eof()) fileExists = true;
    fCheck.close();

    ofstream dataFile(outputFilename, ios::app);
    if (!dataFile.is_open()) return -1;

    // GENERACIÓN DE CABECERA (Header)
    if (!fileExists) {
        for (const auto& loc : activeBits) {
            // Header format: ram_ByteID_bBitID (ej: ram_10_b7)
            dataFile << "ram_" << loc.byteIndex << "_b" << loc.bitOffset << ",";
        }
        dataFile << "action\n";
    } else {
        cout << "Añadiendo datos al dataset existente..." << endl;
    }

    // --- 4. BUCLE DE GRABACIÓN ---
    bool quit = false;
    long frameCount = 0;
    Uint8 *keystates = nullptr;

    while (!alei.game_over() && !quit) {
        SDL_PumpEvents(); 
        keystates = SDL_GetKeyState(NULL);
        if (keystates[SDLK_ESCAPE]) quit = true;

        // Mapeo de Inputs
        bool fire = keystates[SDLK_SPACE];
        bool left = keystates[SDLK_LEFT];
        bool right = keystates[SDLK_RIGHT];
        bool up = keystates[SDLK_UP];
        bool down = keystates[SDLK_DOWN];

        Action currentAction = PLAYER_A_NOOP;

        if (fire) {
            if (right) currentAction = PLAYER_A_RIGHTFIRE;
            else if (left) currentAction = PLAYER_A_LEFTFIRE;
            else if (up) currentAction = PLAYER_A_UPFIRE;
            else if (down) currentAction = PLAYER_A_DOWNFIRE;
            else currentAction = PLAYER_A_FIRE;
        } else {
            if (right) currentAction = PLAYER_A_RIGHT;
            else if (left) currentAction = PLAYER_A_LEFT;
            else if (up) currentAction = PLAYER_A_UP;
            else if (down) currentAction = PLAYER_A_DOWN;
        }

        // SERIALIZACIÓN (BIT A BIT)
        const ALERAM& ram = alei.getRAM();
        
        for (const auto& loc : activeBits) {
            // 1. Obtenemos el byte entero de la RAM
            int byteVal = static_cast<int>(ram.get(loc.byteIndex));
            
            // 2. Extraemos SOLO el bit específico que necesitamos
            // Shift y AND para aislar el bit en loc.bitOffset
            int bitVal = (byteVal >> loc.bitOffset) & 1;
            
            dataFile << bitVal << ",";
        }
        dataFile << currentAction << "\n";

        alei.act(currentAction);
        frameCount++;
        SDL_Delay(10); 
    }

    dataFile.close();
    SDL_Quit();
    cout << "Sesión finalizada. Frames grabados: " << frameCount << endl;
    cout << "Datos guardados en: " << outputFilename << endl;
    return 0;
}