# RA_Proyecto2

Usa make para crear todos los ejecutables necesarios

Al principio habrá que jugar una partida de prueba con ./minimal_agent supported/assault.bin
Una vez se haya jugado esta partida habrá que utilizar ./heatmap para generar la mascara con la que crear el dataset con el que entrenar los distintos modelos
Al crearse la máscara volveremos a ejecutar ./minimal_agent supported/assault.bin y jugaremos las partidas necesarias para crear el dataset

A continuación ya podremos entrenar los modelos con ./bp (Backpropagation) o ./ga <dataset.csv> (Red con Algoritmos Genéticos)
Al ejecutar cualquiera de estas dos opciones necesitaremos especificar el dataset que usará, el entrenado a partir de la máscara se llama dataset_optimized.csv
Si nuestro dataset está desbalanceado por instrucciones en las que no se hace nada podemos usar ./cleaner <dataset.csv> para equilibrarlo, este programa guardará sus resultados en dataset_balanced.csv

Por último para ejecutar el bot que utilice los datos del modelo entrenado usaremos ./ia supported/assault.bin
