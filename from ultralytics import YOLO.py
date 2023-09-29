from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture('C:/Users/Alexey/Desktop/большое светлое будущее/ЛСР/распознавалка номеров/demo.mp4')


model = YOLO('yolov8n.pt')
fps_start_time = time.time()
fps = 0
while True:
    success, img = cap.read()
    results = model(img, stream = True)
    for r in results:
        fps_end_time = time.time()
        fps_diff_time = fps_end_time - fps_start_time
        fps = 1 / fps_diff_time
        fps_start_time = fps_end_time
        fps_text="FPS:{:.2f}".format(fps)
        print(fps_text)
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(img, f'{conf}', (x1, y1-20))
        cvzone.putTextRect(img, f'{fps_text}', (50, 50))
    cv2.imshow("Image", img)
    cv2.waitKey(1)