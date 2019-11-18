import os
import cv2

import random
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import time
import numpy as np


DATADIR = 'raw_data/pet_imgs'
CATEGORIES = ['dog', 'cat']

IMG_SIZE = 80


def create_training_data():
    training_data = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # categories for dog(0) and cat(1)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    return training_data


def save_training_set(training_data):
    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    pickle_out = open('training_set/X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('training_set/y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()


def read_training_set():
    pickle_in = open('training_set/X.pickle', 'rb')
    X = pickle.load(pickle_in)

    pickle_in = open('training_set/y.pickle', 'rb')
    y = pickle.load(pickle_in)

    return X, y


def build_cnn_model():
    name = 'Cats-vs-Dogs-CNN-{}'.format(int(time.time()))

    X, y = read_training_set()

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X / 255.0

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    tensorboard = TensorBoard(log_dir='./output/{}'.format(name))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

    model.save('model/cat_vs_dogs.h5')


def test():
    new_model = tf.keras.models.load_model('model/cat_vs_dogs.h5')

    img_array = cv2.imread('raw_data/pet_imgs/cat/100.jpg', cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    assert new_model.predict(np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)) == 1.0

    img_array = cv2.imread('raw_data/pet_imgs/dog/100.jpg', cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    assert new_model.predict(np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)) == 0.0

    print('Training has completed :-)')


def main():
    training_data = create_training_data()
    random.shuffle(training_data)
    save_training_set(training_data)

    build_cnn_model()

    test()


if __name__ == "__main__":
    main()
