import os

import cv2
import numpy as np

img_size = (70, 70)

names_dict = {0: "The Rock", 1: "Paul", 2: "Vin Diesel"}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture("fast_furious_6_trailer.mp4")

trained_model = cv2.face.LBPHFaceRecognizer_create()
trained_model.read("model.xml")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    tickmark = cv2.getTickCount()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=8, minSize=(img_size)
    )
    for (x, y, w, h) in faces:
        roi_gray = cv2.resize(gray[y : y + h, x : x + w], img_size)
        label, confidence = trained_model.predict(roi_gray)

        cv2.putText(
            frame,
            names_dict[label],
            (x, y - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("image", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    if key == ord("a"):
        for cpt in range(100):
            ret, frame = cap.read()

cv2.destroyAllWindows()
