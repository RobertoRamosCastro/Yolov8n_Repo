from ultralytics import YOLO
import cv2
import cvzone
from sort import *  
from Save_Results_MOC import *

model = YOLO(r'D:\Yolov8n_Repo\YOLO-Course\chapter5-runningYolo\YOLO-Weights\yolov8l.pt')

cap = cv2.VideoCapture(r'D:\Yolov8n_Repo\YOLO-Course\Videos\cars.mp4')
#cap = cv2.VideoCapture(r'D:\Yolov8n_Repo\YOLO-Course\Videos\test3.mp4')


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

#deepsort = DeepSort(model_path=r"D:\Yolov8n_Repo\YOLO-Course\yolov8_deepsort_tracking\YOLOv8-DeepSORT-Object-Tracking\ultralytics\yolo\v8\detect\deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt.t7")

    
# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Objetos permitidos
allowed_objects = ['truck', 'car', 'bus', 'motorbike']

# Creating the line to count objects
limits = [400,297,673,297] # Values of the mask
#limits = [220,450,600,450] #x1,y1,x2,y2 -- (0,0) top left 

# contador para los coches que cruzan la linea
counter_cars = 0
counter_mb = 0
counter_pesados = 0
dict_with_results = {}

# lista de id's ya contados
list_id_counted = []
outputs = []
oids = []
bboxes = []
confidences = []

while True:
    ret, frame = cap.read()
    #print('frame.shape',frame.shape)

    if not ret:
        break

    results = model(frame, stream=True)

    # Para inicializar nuestros tracker
    detections=np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            #x1,y1,x2,y2 = box.xyxy[0]
            #print('bbox',x1,y1,x2,y2)
            x1,y1,x2,y2 = [int(val) for val in box.xyxy[0]]
            w, h = x2-x1, y2-y1
            # Confidence
            conf = box.conf[0]
            #conf = round(float(conf),2) #tambien puedo ponerle dos deicamales con 'conf:.2f'
            conf = float(conf)
            # Class Names
            cls = int(box.cls[0])
            # print(conf,classNames[cls])

            # Clases de interes
            if classNames[cls] in allowed_objects:
                if conf > 0.3:
                # Mostrar bounding box
                # cvzone.cornerRect(frame, bbox=(x1,y1,w,h), l=15, t=2, rt=5)
                # Mostrarlo en texto
                    '''cv2.putText(frame,
                                f'{classNames[cls]}',
                                (max(0, x1), max(35, y1)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                thickness=1, 
                                fontScale=1,
                                color=(0, 255, 0)
                                )'''
                # adding new detections in each loop
                '''bbox = [x1,y1,x2,y2]
                bboxes.append(bbox)
                oids.append(cls)
                confidences.append(conf)'''
                currentArray = np.array([x1,y1,x2,y2,cls])
                detections = np.vstack((detections, currentArray))

    #print('dets', detections)
    # update with a list of detections
    resultTracker = tracker.update(detections)
    #outputs = deepsort.update(bbox_xywh=bboxes, confidences=confidences, oids=oids, ori_img=frame)

    cv2.line(frame, (limits[0],limits[1]),(limits[2]  ,limits[3]), (0,0,255), 5)

    for result in resultTracker:
        #print('result',result)
        x1,y1,x2,y2,c,id= [int(val) for val in result]
        #x1,y1,x2,y2,id= [int(val) for val in result]
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(frame, bbox=(x1,y1,w,h), l=15, t=2, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame,
                            f'{id} {classNames[c]}',
                            (max(0,x1), max(35, y1)),
                            scale=1,
                            thickness=1,
                            offset=3) 
        
        cx, cy =  x1+w//2, y1+h//2
        cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if id not in list_id_counted:
                if classNames[c] == 'car':
                    counter_cars+=1
                elif classNames[c] == 'motorbike':
                    counter_mb+=1    
                elif classNames[c] == 'bus' or classNames[c] == 'truck':
                    counter_pesados +=1
                list_id_counted.append(id)
                # Cuando detecta uno cambia de color
                cv2.line(frame, (limits[0],limits[1]),(limits[2],limits[3]), (0,255,0), 5)

    # Mostar el conteo total
    cv2.putText(frame, f'Coches: {counter_cars}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    dict_with_results['Coches'] = counter_cars
    cv2.putText(frame, f'Veh.Ligeros: {counter_mb}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    dict_with_results['Veh.Ligeros'] = counter_mb
    cv2.putText(frame, f'Veh.Pesados: {counter_pesados}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    dict_with_results['Veh.Pesados'] = counter_pesados
    
    cv2.imshow("Frame", frame)

    # Esperar a que se presione la tecla 'q' para cerrar la ventana
    key = cv2.waitKey(1)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

crear_Tabla_csv(dict_with_results)


