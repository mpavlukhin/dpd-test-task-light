import os
import cv2

import random
import pickle


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


def main():
    training_data = create_training_data()
    random.shuffle(training_data)
    save_training_set(training_data)


if __name__ == "__main__":
    main()
