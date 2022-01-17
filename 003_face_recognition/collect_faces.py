import os

import cv2
import numpy as np

folder_path = "./raw_images/"
save_directory = "train_faces/"

current_id = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for root, dirs, files in os.walk(folder_path):
    print(root, dirs, files)
    if len(files):
        for file in files:
            file_path = root + "/" + file
            print("working with file: ", file_path)
            img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            img_grey = np.array(img_grey, dtype="uint8")
            faces = face_cascade.detectMultiScale(
                img_grey, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30)
            )
            print(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(img_grey, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = img_grey[y : y + h, x : x + w]

                save_folder = str.split(root, "/")[-1]
                save_path = (
                    save_directory + save_folder + "/" + str(current_id) + ".jpg"
                )
                status = cv2.imwrite(save_path, roi,)
                print("Saving to: ", save_path, " status: ", status)

                current_id += 1
