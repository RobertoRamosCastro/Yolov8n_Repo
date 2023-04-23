import re

nombre_archivo = r'D:\Yolov8n_Repo\YOLO-Course\TFG\tabla_resultados.csv'

def crear_Tabla_csv(diccionario):
    with open(nombre_archivo, 'w') as f:
        for key, value in diccionario.items():
            f.write('{}:{},'.format(key, value))


def sumar_clases_tabla():
    value_total = 0
    with open(nombre_archivo, 'r') as f:
        contenido = f.read()
        contenido_split1 = re.split(':|,|\n', contenido)
        contenido_split1.pop()
        if len(contenido_split1) == 6:
            for value in range(len(contenido_split1)):
                if value % 2 != 0:
                    value_total += int(contenido_split1[value])
            porcentajeAcierto(value_total)
        else:
            print('No estan todas las clases.')

def porcentajeAcierto(total_OC):
    # precision = verdaderos positivos / (verdaderos positivos + falsos positivos)
    verdad_Pos = 72 # conteo manual
    falsos_Pos = abs(verdad_Pos-total_OC)
    precision = verdad_Pos / (verdad_Pos + falsos_Pos)
    print(round(precision,2),'% de precision en conteo del modelo')
