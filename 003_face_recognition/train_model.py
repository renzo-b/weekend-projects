import os

import cv2
import numpy as np

folder_path = "./train_faces/"

x_train = []
y_train = []
label_ids = {}  # "ID":"label"

img_size = (70, 70)
current_label = 0
current_id = 0  # ID for every image


def process_image(img, img_size):
    img = cv2.resize(img, img_size)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.Laplacian(img, cv2.CV_64F, 5)
    return img


for root, dirs, files in os.walk(folder_path):
    print(root, dirs, files)
    if len(files):
        for i, file in enumerate(files):
            file_path = root + "/" + file
            print("working with file: ", file_path)
            img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_processed = process_image(img_grey, img_size)

            # while True:
            #     cv2.imshow("img", img_processed)
            #     if cv2.waitKey(1) == ord("q"):
            #         break

            x_train.append(img_processed)
            y_train.append(current_label)
            label_ids[current_id] = current_label

            current_id += 1
            if i == len(files) - 1:
                current_label += 1

cv2.destroyAllWindows()

print("IDs: ", label_ids)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(x_train, y_train)

model.save("model.xml")
