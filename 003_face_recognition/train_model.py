import os
import pickle

import cv2
import numpy as np

folder_path = "./images/"

x_train = []
y_train = []
label_ids = {}  # "ID":"label"


current_label = 0
current_id = 0  # ID for every image

for root, dirs, files in os.walk(folder_path):
    print(root, dirs, files)
    if len(files):
        for i, file in enumerate(files):
            file_path = root + "/" + file
            print("working with file: ", file_path)
            img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_blur = cv2.GaussianBlur(img_grey, (5, 5), 0)
            img_laplace = cv2.Laplacian(img_blur, cv2.CV_64F, 5)

            while True:
                cv2.imshow("img", img_laplace)
                if cv2.waitKey(1) == ord("q"):
                    break
            x_train.append(img_laplace)
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

## test
folder_path = "./test_images/"

x_test = []
y_test = []
label_ids = {}  # "ID":"label"


current_label = 0
current_id = 0  # ID for every image

for root, dirs, files in os.walk(folder_path):
    print(root, dirs, files)
    if len(files):
        for i, file in enumerate(files):
            file_path = root + "/" + file
            print("working with file: ", file_path)
            img_grey = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            x_test.append(img_grey)
            y_test.append(current_label)
            label_ids[current_id] = current_label

            current_id += 1
            if i == len(files) - 1:
                current_label += 1

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

for i, x in enumerate(x_test):
    prediction = model.predict(x)
    print("prediction: ", prediction, "actual: ", y_test[i])
