import cv2
import numpy as np
from ultralytics import YOLO
import time
import random
from pyzbar import pyzbar

VIDEO_PATH = "test.mov"
CONF = 0.4
MAP_SIZE = 150

PROC_W = 960
PROC_H = 540

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

medical_active = False
medical_time = 0
surveillance_id = -1

soldier_stats = {}

# Simulated Metrics

def generate_stats(id):
    if id not in soldier_stats:
        soldier_stats[id] = {
            "temp": round(random.uniform(36.2, 39.0), 1),
            "hr": random.randint(70, 130),
            "health": random.randint(60, 100)
        }
    return soldier_stats[id]


# Threat Level 

def threat_color(stats):
    if stats["health"] < 70 or stats["hr"] > 115:
        return "HIGH", (0,0,255)
    elif stats["health"] < 85 or stats["hr"] > 95:
        return "MEDIUM", (0,255,255)
    else:
        return "LOW", (0,255,0)


# QR Detection

def detect_qr(frame):
    decoded = pyzbar.decode(frame)
    enemies = []
    for qr in decoded:
        x,y,w,h = qr.rect
        enemies.append((x,y,x+w,y+h))
    return enemies


while True:

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue

    frame = cv2.resize(frame,(PROC_W,PROC_H))
    h,w,_ = frame.shape

    qr_enemies = detect_qr(frame)
    results = model(frame,conf=CONF)[0]

    persons=[]
    boxes=[]

    if results.boxes is not None:
        for box in results.boxes:
            if int(box.cls[0])==0:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                persons.append(((x1+x2)//2,(y1+y2)//2))
                boxes.append((x1,y1,x2,y2))

    # Draw Soldiers
    for i,(x1,y1,x2,y2) in enumerate(boxes):

        stats = generate_stats(i)
        threat, color = threat_color(stats)

        thick = 3 if i==surveillance_id else 2
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,thick)

        label = f"{threat} | T:{stats['temp']} HR:{stats['hr']} HP:{stats['health']}%"
        cv2.putText(frame,label,(x1,y2+18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        if i==surveillance_id:
            cv2.putText(frame,"SURVEILLANCE TARGET",
                        (x1,y1-25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.putText(frame,f"Soldier {i}",
                    (x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    # QR Enemies
    for ex1,ey1,ex2,ey2 in qr_enemies:
        cv2.rectangle(frame,(ex1,ey1),(ex2,ey2),(255,0,255),2)
        cv2.putText(frame,"QR ENEMY",
                    (ex1,ey1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)

    # RADAR
    radar=np.zeros((MAP_SIZE,MAP_SIZE,3),dtype=np.uint8)
    for p in persons:
        mx=int(p[0]/w*MAP_SIZE)
        my=int(p[1]/h*MAP_SIZE)
        cv2.circle(radar,(mx,my),3,(0,0,255),-1)

    cv2.rectangle(radar,(0,0),(MAP_SIZE-1,MAP_SIZE-1),(0,255,0),1)
    frame[10:10+MAP_SIZE,10:10+MAP_SIZE]=radar

    #minimap
    cv2.putText(frame,f"ENEMIES: {len(qr_enemies)}",
                (20+MAP_SIZE+10,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    
    if medical_active:
        if time.time() - medical_time < 5:
            cv2.putText(frame,"MEDICAL DRONE ACTIVE",
                        (w//2-170,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),3)
        else:
            medical_active = False
    

    cv2.imshow("AI Combat Drone",frame)

    key=cv2.waitKey(1)&0xFF

    if key==ord('a'):
        medical_active=True
        medical_time=time.time()

    if key==ord('s') and persons:
        surveillance_id+=1
        if surveillance_id>=len(persons):
            surveillance_id=0

    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
