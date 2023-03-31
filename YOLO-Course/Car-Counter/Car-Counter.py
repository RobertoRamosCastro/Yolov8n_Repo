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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    imgRegion = cv2.bitwise_and(frame, mask) # operacion para extraer la parte que queremos con nuestra mascara

    results = model(frame, stream=True)
    # Para inicializar nuestros tracker
    detections=np.empty((0,5))

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
            if classNames[cls] == 'car' or classNames[cls] == 'truck' or classNames[cls]=='motorbike' or classNames[cls]=='bus':
                if conf > 0.5:
                    # Mostrar bounding box
                    cvzone.cornerRect(frame, bbox=(x1,y1,w,h), l=15, t=2, rt=5)
                    # Mostrarlo en texto
                    '''cvzone.putTextRect(frame,
                                    f'{classNames[cls]} {conf:.2f} {ID}',
                                    (max(0,x1), max(35, y1)),
                                    scale=1,
                                    thickness=1,
                                    offset=3) '''
                    # adding new detections in each loop
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections, currentArray))


    # update with a list of detections
    resultTracker = tracker.update(detections)

    for result in resultTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(frame, bbox=(x1,y1,w,h), l=15, t=2, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(frame,
                                    f'{id}',
                                    (max(0,x1), max(35, y1)),
                                    scale=1,
                                    thickness=1,
                                    offset=3) 

    
    cv2.imshow("Frame", frame)
    cv2.imshow("ImgRegion", imgRegion)

    # Esperar a que se presione la tecla 'q' para cerrar la ventana y 50 ms entre frames
    key = cv2.waitKey(0)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()



