# -*- coding: utf-8 -*-
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.models import Sequential
from keras.layers import Dense, Lambda, Cropping2D, Conv2D, Dense, Activation, MaxPooling2D, Flatten
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

lines = []

left_correction = 0.2
right_correction = 0.2

path = "D:/17.Behavioral_Cloning_Project/data/data/driving_log.csv"

with open(path) as input:
    reader = csv.reader(input)
    for line in reader:
        lines.append(line)
    lines.pop(0)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            center_images = []
            measurements = []
            for line in batch_samples:
                for i in range(3):
                    path = "D:/17.Behavioral_Cloning_Project/data/data/" + (line[i].replace(" ", ""))
                    image = imread(path)
                    if i == 0:
                        measurement = float(line[3])
                    elif i == 1:
                        measurement = float(line[3]) + left_correction
                    elif i == 2:
                        measurement = float(line[3]) - right_correction
                    # 添加正常数据
                    center_images.append(image)
                    measurements.append(measurement)
                    # 添加翻转数据
                    center_images.append(np.fliplr(image))
                    measurements.append(-measurement)

            X_train = np.array(center_images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)


# (X_train, y_train) = generator(train_samples, batch_size=32)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

train_steps = np.ceil(len(train_samples) / 32).astype(np.int32)
validation_steps = np.ceil(len(validation_samples) / 32).astype(np.int32)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator,
                    steps_per_epoch=train_steps,
                    epochs=5,
                    validation_data=validation_generator,
                    validation_steps=validation_steps)
model.save('model.h5')
