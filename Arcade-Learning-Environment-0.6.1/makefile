# Compilador
CXX = g++

# Flags para ALE y SDL (Librerías y Rpath)
ALE_FLAGS = -L. -lSDL -lale -Wl,-rpath=.

# Flags de optimización para los entrenadores
OPT_FLAGS = -O3

# --- OBJETIVOS ---

# Escribe 'make' o 'make all' para compilar todo
all: minimal_agent bp ga ia heatmap cleaner

minimal_agent: minimal_agent.cpp
	$(CXX) minimal_agent.cpp -o minimal_agent $(ALE_FLAGS)

bp: bp.cpp
	$(CXX) bp.cpp -o bp $(OPT_FLAGS)

ga: ga.cpp
	$(CXX) ga.cpp -o ga $(OPT_FLAGS) $(ALE_FLAGS)

ia: ia.cpp
	$(CXX) ia.cpp -o ia $(ALE_FLAGS)

heatmap: heatmap.cpp
	$(CXX) heatmap.cpp -o heatmap $(OPT_FLAGS)

cleaner: cleaner.cpp
	$(CXX) cleaner.cpp -o cleaner $(OPT_FLAGS)

clean:
	rm -f minimal_agent bp ga ia heatmap cleaner