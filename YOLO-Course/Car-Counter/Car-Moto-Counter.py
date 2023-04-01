from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *  

model = YOLO(r'D:\Yolov8n_Repo\YOLO-Course\chapter5-runningYolo\YOLO-Weights\yolov8n.pt')

cap = cv2.VideoCapture(r'D:\Yolov8n_Repo\YOLO-Course\Videos\cars.mp4')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread(r'D:\Yolov8n_Repo\YOLO-Course\Car-Counter\mascara.png')

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Creating the line to count objects
limits = [400,297,673,297] # Values of the mask

# contador para los coches que cruzan la linea
counter = 0

# lista de id's ya contados
list_id_counted_cars = []
list_id_counted_motos = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    imgRegion = cv2.bitwise_and(frame, mask) # operacion para extraer la parte que queremos con nuestra mascara

    results = model(frame, stream=True)

    # Para inicializar nuestros tracker
    detections_cars=np.empty((0,5))
    detections_motos=np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            # Confidence
            conf = box.conf[0]
            #conf = round(float(conf),2) #tambien puedo ponerle dos deicamales con 'conf:.2f'
            conf = float(conf)
            # Class Names
            cls = int(box.cls[0])
            # print(conf,classNames[cls])

            # Clases de interes
            if classNames[cls] == 'car':
                #if conf > 0.3:
                # Mostrar bounding box
                # cvzone.cornerRect(frame, bbox=(x1,y1,w,h), l=15, t=2, rt=5)
                # Mostrarlo en texto
                '''cvzone.putTextRect(frame,
                                f'{classNames[cls]} {conf:.2f} {ID}',
                                (max(0,x1), max(35, y1)),
                                scale=1,
                                thickness=1,
                                offset=3) '''
                # adding new detections in each loop
                currentArray_car = np.array([x1,y1,x2,y2,conf])
                detections_cars = np.vstack((detections_cars, currentArray_car))

                
            elif classNames[cls]=='motorbike':
                currentArray_moto = np.array([x1,y1,x2,y2,conf])
                detections_motos = np.vstack((detections_motos, currentArray_moto))

    # update with a list of detections
    resultTracker_car = tracker.update(detections_cars)
    resultTracker_moto = tracker.update(detections_motos)

    cv2.line(frame, (limits[0],limits[1]),(limits[2],limits[3]), (0,0,255), 5)

    for result in resultTracker_car:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2,id = int(x1), int(y1), int(x2), int(y2), int(id)
        # print(x1,y1,x2,y2,id)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(frame, bbox=(x1,y1,w,h), l=15, t=2, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame,
                            f'{id} {classNames[cls]}',
                            (max(0,x1), max(35, y1)),
                            scale=1,
                            thickness=1,
                            offset=3) 
        
        cx, cy =  x1+w//2, y1+h//2
        cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if id not in list_id_counted_cars:
                counter+=1
                list_id_counted_cars.append(id)
                # Cuando detecta uno cambia de color
                cv2.line(frame, (limits[0],limits[1]),(limits[2],limits[3]), (0,255,0), 5)

    for result in resultTracker_moto:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2,id = int(x1), int(y1), int(x2), int(y2), int(id)
        # print(x1,y1,x2,y2,id)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(frame, bbox=(x1,y1,w,h), l=15, t=2, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame,
                            f'{id} {classNames[cls]}',
                            (max(0,x1), max(35, y1)),
                            scale=1,
                            thickness=1,
                            offset=3) 
        
        cx, cy =  x1+w//2, y1+h//2
        cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if id not in list_id_counted_motos:
                counter+=1
                list_id_counted_motos.append(id)
                # Cuando detecta uno cambia de color
                cv2.line(frame, (limits[0],limits[1]),(limits[2],limits[3]), (0,255,0), 5)

    # Mostar el conteo total
    cvzone.putTextRect(frame, f'Counts: {len(list_id_counted_cars)}', (50,50)) 
    cvzone.putTextRect(frame, f'Counts: {len(list_id_counted_motos)}', (150,150)) 
    
    cv2.imshow("Frame", frame)
    # cv2.imshow("ImgRegion", imgRegion)

    # Esperar a que se presione la tecla 'q' para cerrar la ventana
    key = cv2.waitKey(0)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()



