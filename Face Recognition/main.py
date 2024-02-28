import cv2
import numpy as np
import face_recognition
import os


path = 'Images'
images = []
pplNames = []
myList = os.listdir(path)
for ppl in myList:
    curImg = cv2.imread(f'{path}/{ppl}')
    images.append(curImg)
    pplNames.append(os.path.splitext(ppl)[0])



def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = pplNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 -25), (x2, y2), (255, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1,y2-5), cv2.FONT_ITALIC, 0.7, (0, 0, 0), 1)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)