import cv2
import os

import tensorflow as tf
import numpy as np

from cnn.train import IMG_SIZE


def predict(filepath):
    basedir = os.path.abspath(os.path.dirname(__file__))
    new_model = tf.keras.models.load_model(os.path.join(basedir, 'model/cat_vs_dogs.h5'))

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    result = new_model.predict(np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1))

    return 'Cat' if result == 1.0 else 'Dog'
