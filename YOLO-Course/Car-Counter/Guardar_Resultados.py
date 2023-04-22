import re

nombre_archivo = 'tabla_resultados.csv'

def crear_Tabla_csv(clase_vehiculo, len_list_id):
    try:
        with open(nombre_archivo, 'r') as f:
            counters = f.read()
            if not ('Moto' in counters or 'Coche' in counters or 'Bus' in counters or 'Camion' in counters):
                with open(nombre_archivo, 'a') as f:
                    f.write('Moto:0,Coche:0,Bus:0,Camion:0\n')
    except FileNotFoundError:
        with open(nombre_archivo, 'w') as f:
            f.write('Moto:0,Coche:0,Bus:0,Camion:0\n')

    with open(nombre_archivo, 'r') as f:
        counters = f.read()
    if clase_vehiculo in counters:
        pos_ini = counters.index(clase_vehiculo) + len(clase_vehiculo) + 1
        pos_fin = counters.index(',', pos_ini)
        with open(nombre_archivo, 'w') as f:
            f.write(counters[:pos_ini] + str(len_list_id) + counters[pos_fin:])
    else:
        with open(nombre_archivo, 'a') as f:
            f.write(clase_vehiculo + ':' + str(len_list_id) + ',')


def sumar_clases_tabla():
    value_total = 0
    with open(nombre_archivo, 'r') as f:
        contenido = f.read()
        contenido_split1 = re.split(':|,|\n', contenido)
        mi_array = contenido_split1[8:]
        mi_array.pop()
        if len(mi_array) == 8:
            for value in range(len(mi_array)):
                if value % 2 != 0:
                    value_total += int(mi_array[value])
            porcentajeAcierto(value_total)
            #print(value_total)
        else:
            print('No estan todas las clases.')

def porcentajeAcierto(total_OC):
    # precision = verdaderos positivos / (verdaderos positivos + falsos positivos)
    verdad_Pos = 72 # conteo manual
    falsos_Pos = abs(verdad_Pos-total_OC)
    precision = verdad_Pos / (verdad_Pos + falsos_Pos)
    print(precision,'% de precision en conteo del modelo')
