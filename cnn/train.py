import os
import cv2


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


def main():
    training_data = create_training_data()
    print(len(training_data))


if __name__ == "__main__":
    main()
