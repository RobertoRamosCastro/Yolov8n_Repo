import cv2
import numpy as np
import urllib.request

'''# Descargar el archivo .cfg de YOLOv3 desde GitHub
url_cfg = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
response_cfg = urllib.request.urlopen(url_cfg)
text_cfg = response_cfg.read().decode('utf-8')
with open('yolov3.cfg', 'w') as f:
    f.write(text_cfg)

# Descargar el archivo de pesos de YOLOv3 desde GitHub
url_weights = 'https://pjreddie.com/media/files/yolov3.weights'
response_weights = urllib.request.urlopen(url_weights)
weights = np.array(bytearray(response_weights.read()), dtype=np.uint8)
with open('yolov3.weights', 'wb') as f:
    f.write(weights)

# Descargar archivo con coco names
url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
filename = "coco.names"
urllib.request.urlretrieve(url, filename)'''


# Cargar modelo de detección de objetos
net = cv2.dnn.readNetFromDarknet(r'D:\Yolov8n_Repo\YOLO-Course\MOC\yolov3.cfg', r'D:\Yolov8n_Repo\YOLO-Course\MOC\yolov3.weights')

# Cargar clases
with open(r'D:\Yolov8n_Repo\YOLO-Course\MOC\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Configurar líneas de conteo
line_positions = [(0, 300), (800, 300)] # posición de las dos líneas
line_thickness = 2
line_color = (0, 255, 0)

# Inicializar contadores de objetos
object_counts = {class_name: 0 for class_name in classes}

# Inicializar objeto de seguimiento de objetos
tracker = cv2.legacy.MultiTracker_create()

# Capturar video de la cámara
cap = cv2.VideoCapture(r'D:\Yolov8n_Repo\YOLO-Course\Videos\cars.mp4')

while True:
    # Leer frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar detección de objetos
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    
    # Procesar detecciones
    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Solo considerar detecciones con confianza mayor a 0.5
        if confidence > 0.5:
            class_name = classes[class_id]
            
            # Extraer bounding box de la detección
            box = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")
            
            # Inicializar tracker para el objeto
            tracker.add(cv2.TrackerKCF_create(), frame, (x, y, w, h))
            
            # Dibujar bounding box y etiqueta en el frame
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{class_name} {object_counts[class_name]}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Actualizar trackers y contar objetos que cruzan la línea
    ok, boxes = tracker.update(frame)
    for box in boxes:
        x, y, w, h = [int(v) for v in box]
        cx, cy = x + w//2, y + h//2
        
        # Comprobar si objeto cruza alguna de las líneas de conteo
        for i, (p1, p2) in enumerate(zip(line_positions[:-1], line_positions[1:])):
            if p1[1] < cy < p2[1] or p2[1] < cy < p1[1]:
                cv2.line(frame, p1, p2, line_color, line_thickness)
                
                # Incrementar contador para la clase correspondiente
                class_name = classes[class_id]
                object_counts[class_name] += 1
    
       # Mostrar frame con resultados
    cv2.imshow('Object Counting', frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Imprimir resultados
print('Object counts:')
for class_name, count in object_counts.items():
    print(f'{class_name}: {count}')


