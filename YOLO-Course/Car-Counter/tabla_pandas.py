import pandas as pd

def crear_Tabla(archivo_txt):
    # Leer los resultados de counters.txt y almacenarlos en un DataFrame de pandas
    df = pd.read_csv(archivo_txt, names=['Script 1', 'Script 2', 'Script 3', 'Script 4'])

    # Agregar títulos a las columnas
    df.columns.name = 'Scripts'
    df.index.name = 'Clases'

    # Mostrar la tabla en la consola
    print(df)


variable = 15
with open('tabla.csv', 'a') as f:
    f.write(str(variable) + '\n')