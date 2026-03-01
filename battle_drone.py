import cv2
import numpy as np
from ultralytics import YOLO
import time

VIDEO_PATH = "test.mov"
CONF = 0.4
MAP_SIZE = 200

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Camera not found!")
    exit()

print("AI Battle Drone Started")

medical_active = False
marked_target = None
medical_time = 0

while True:

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue

    h,w,_ = frame.shape

    results = model(frame,conf=CONF)[0]

    persons = []
    boxes = []

    if results.boxes is not None:
        for box in results.boxes:
            cls=int(box.cls[0])
            if cls==0:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                cx=(x1+x2)//2
                cy=(y1+y2)//2
                persons.append((cx,cy))
                boxes.append((x1,y1,x2,y2))

    #  auto lock
    locked=None
    if persons:
        center=(w//2,h//2)
        locked=min(persons,key=lambda p:(p[0]-center[0])**2+(p[1]-center[1])**2)

    # Draw detections
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        color=(0,0,255)
        if marked_target==i:
            color=(0,255,255)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"Enemy {i}",(x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    # auto lock line
    if locked:
        cv2.line(frame,(w//2,h//2),locked,(255,0,0),2)
        cv2.putText(frame,"AUTO LOCK",(locked[0]-20,locked[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    # Mini map
    radar=np.zeros((MAP_SIZE,MAP_SIZE,3),dtype=np.uint8)
    for p in persons:
        mx=int(p[0]/w*MAP_SIZE)
        my=int(p[1]/h*MAP_SIZE)
        cv2.circle(radar,(mx,my),4,(0,0,255),-1)

    cv2.rectangle(radar,(0,0),(MAP_SIZE-1,MAP_SIZE-1),(0,255,0),1)
    frame[10:10+MAP_SIZE,10:10+MAP_SIZE]=radar

    # Medical assistance
    if medical_active:
        if time.time()-medical_time<5:
            cv2.putText(frame,"MEDICAL DRONE DEPLOYED",(200,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        else:
            medical_active=False

    # Dashboard
    cv2.putText(frame,f"Enemies Detected: {len(persons)}",(20,240),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.putText(frame,"A = Medical Assist",(20,270),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    cv2.putText(frame,"M = Mark Closest Target",(20,300),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    cv2.putText(frame,"ESC = Quit",(20,330),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

    cv2.imshow("AI Combat Drone",frame)

    key=cv2.waitKey(1)&0xFF

    # Medical Assist
    if key==ord('a'):
        medical_active=True
        medical_time=time.time()
        print("Medical Assistance Activated")

    # Mark Closest Target
    if key==ord('m') and persons:
        marked_target=0
        print("Target Marked")

    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
