import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


def load_images(train_dir, test_dir, train_label, test_label):
    train_imgs = os.listdir(train_dir)
    test_imgs = os.listdir(test_dir)

    train_data = []
    test_data = []
    for name in train_imgs:
        img_path = os.path.join(train_dir, name)
        img = cv2.imread(img_path)
        train_data.append(img)

    for name in test_imgs:
        img_path = os.path.join(test_dir, name)
        img = cv2.imread(img_path)
        test_data.append(img)

    train_data = np.array(train_data)/255
    test_data = np.array(test_data)/255

    train_label = pd.read_csv(train_label)["filename", "box_width", "box_height", "angle"]
    test_label = pd.read_csv(test_label)
    print(train_label)


if __name__ == "__main__":
    load_images("data/train", "data/test", "data/train(41-200).csv", "data/evaluate(0-40).csv")