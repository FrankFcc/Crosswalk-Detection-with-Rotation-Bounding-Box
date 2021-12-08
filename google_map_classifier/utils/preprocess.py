import numpy as np
import tensorflow as tf
import os
import cv2
import pandas as pd


def get_data(image_path, label_file):
    """
    Read in image datas and find their labels
    :param image_path: Folder directory containing all images
    :param label_file: Label for the images
    :return:
    """
    image_names = os.listdir(image_path)
    images = []
    labels = pd.read_csv(label_file, usecols=[1], names=["class"]).to_numpy().squeeze(1)
    read_labels = []
    for name in image_names:
        index = int(name.split('.')[0])
        # The images are not always read in the correct numerical order
        read_labels.append(labels[index])
        img_path = os.path.join(image_path, name)
        img = cv2.resize(cv2.imread(img_path), (320, 320))
        images.append(img)
    labels = tf.one_hot(read_labels, 2)
    return np.array(images)/255, labels


if __name__ == "__main__":
    print(get_data("../images/all_images", "../images/label.csv"))
