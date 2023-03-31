from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO(r'D:\YOLO-Course\chapter5-runningYolo\YOLO-Weights\yolov8n.pt')

cap = cv2.VideoCapture(r'D:\YOLO-Course\Videos\motorbikes.mp4')

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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, stream=True)

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
                # Mostrar bounding box
                cvzone.cornerRect(frame, bbox=(x1,y1,w,h), t=2)
                # Mostrarlo en texto
                cvzone.putTextRect(frame,
                                f'{classNames[cls]} {conf:.2f}',
                                (max(0,x1), max(35, y1)),
                                scale=1,
                                thickness=1)
            else:
                continue

            

    cv2.imshow("Frame", frame)

    # Esperar a que se presione la tecla 'q' para cerrar la ventana y 50 ms entre frames
    key = cv2.waitKey(0)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()



