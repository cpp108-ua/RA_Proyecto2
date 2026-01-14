#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include "src/ale_interface.hpp"
#include <SDL/SDL.h> 

using namespace std;

void usage(char const* pname) {
    cerr << "\nUSAGE:\n" << "    " << pname << " <romfile>\n";
    exit(-1);
}

int main(int argc, char **argv) {
    if (argc < 2) usage(argv[0]);

    // --- CARGA DE MÁSCARA ---
    vector<int> activeBytes;
    ifstream maskFile("mask.txt");
    bool hasMask = maskFile.is_open(); // Detectamos si existe la máscara

    if (hasMask) {
        int byteIndex;
        while (maskFile >> byteIndex) {
            if (byteIndex >= 0 && byteIndex < 128) activeBytes.push_back(byteIndex);
        }
        maskFile.close();
        cout << "Modo Optimizado: Grabando " << activeBytes.size() << " bytes." << endl;
    } else {
        cout << "Modo Completo (Sin mascara): Grabando 128 bytes." << endl;
        for(int i=0; i<128; ++i) activeBytes.push_back(i);
    }

    string outputFilename = hasMask ? "dataset_optimized.csv" : "dataset.csv";
    
    cout << "Guardando datos en: " << outputFilename << endl;

    // --- INICIALIZACIÓN DE ENTORNO ---
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) < 0) return -1;

    ALEInterface alei{};
    alei.setInt("random_seed", 123);
    alei.setBool("display_screen", true); 
    alei.setBool("sound", true);
    alei.loadROM(argv[1]);
    
    // --- PREPARACIÓN DE DATASET ---
    bool fileExists = false;
    ifstream fCheck(outputFilename);
    if (fCheck.good() && fCheck.peek() != ifstream::traits_type::eof()) fileExists = true;
    fCheck.close();

    ofstream dataFile(outputFilename, ios::app);
    if (!dataFile.is_open()) return -1;

    if (!fileExists) {
        for (int idx : activeBytes) dataFile << "ram_" << idx << ",";
        dataFile << "action\n";
    } else {
        cout << "Añadiendo datos al dataset existente..." << endl;
    }

    // --- BUCLE DE GRABACIÓN ---
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

        // Serialización
        const ALERAM& ram = alei.getRAM();
        for (int idx : activeBytes) {
            dataFile << static_cast<int>(ram.get(idx)) << ",";
        }
        dataFile << currentAction << "\n";

        alei.act(currentAction);
        frameCount++;
        SDL_Delay(10); 
    }

    dataFile.close();
    SDL_Quit();
    cout << "Sesión finalizada. Frames grabados: " << frameCount << endl;
    return 0;
}