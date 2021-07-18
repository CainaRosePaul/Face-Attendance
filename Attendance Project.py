import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
image=[]
classname=[]
path='Image Attendance'
classList=os.listdir(path)
print(classList)

for cl in classList:
    curImg=cv2.imread(f'{path}/{cl}')
    image.append(curImg)
    classname.append(os.path.splitext(cl)[0])
    
    
#print(classname)

def encoding(image):
    encodeList = []
    for img in image:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markattendence(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')

encodeListKnown=encoding(image)
print(classname)

cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #resize to 1/4
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    
    faceCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)
        #print(matchIndex)
        if matches[matchIndex]:
            name=classname[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
            markattendence(name)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    
    
    
    