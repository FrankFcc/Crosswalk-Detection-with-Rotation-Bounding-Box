import math

import numpy as np
import tensorflow as tf
import cv2
import requests
import math


class MapReader:
    def __init__(self):
        self.classifier = tf.keras.models.load_model("saved_model/")
        self.url = "https://maps.googleapis.com/maps/api/staticmap?"

    def classify(self, img):
        """
        Classify whether an image is of a city or not using a pretrained resnet-like model
        :param img: input image
        :return: prediction 1:city, 0:not city
        """
        prediction = np.argmax(self.classifier(img))
        return prediction

    def get_image(self, lat, long, zoom, api_key, filename):
        """
        Download a image from google map api given location
        :param lat: latitude
        :param long: longitude
        :param zoom: zoom level
        :param api_key: Your api key for google map api
        :param filename: Complete directory to save the file
        :return: None
        """

        center = str(lat) + ',' + str(long)
        r = requests.get(self.url + "center=" + center + "&zoom=" + str(zoom)
                         + "&size=640x640" + "&maptype=satellite" + "&key="
                         + api_key)
        f = open(filename, 'wb')
        f.write(r.content)
        f.close()

    def get_continuous_image(self, lat, long, zoom, api_key, folder_name, horizontal_steps, vertical_steps):
        """
        Download continuous images from google map api
        :param lat: latitude
        :param long: longitude
        :param zoom: zoom level
        :param api_key: Your api key for google map api
        :param folder_name: directory to save the files in
        :param horizontal_steps: how many pictures to download horizontally for each vertical step
        :param vertical_steps: how many pictures to download vertically for each horizontal step
        :return: None
        """
        px, py = self.latlon2pixels(lat, long, zoom)
        xs = []
        ys = []
        for i in range(horizontal_steps):
            xs.append(px+i*640)
        for i in range(vertical_steps):
            ys.append(py+i*640)
        centers = []
        for y in ys:
            for x in xs:
                centers.append(self.pixels2latlon(x, y, zoom))

        for i, center in enumerate(centers):
            r = requests.get(self.url + "center=" + str(center[0]) + ',' + str(center[1]) + "&zoom=" + str(zoom)
                             + "&size=640x640" + "&maptype=satellite" + "&key="
                             + api_key)
            f = open(folder_name + str(i) + '.png', 'wb')
            f.write(r.content)
            f.close()

    def latlon2pixels(self, lat, lon, zoom):
        """
        Converts latitude and longitude to pixel locations
        original author: https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser/50536888#50536888
        :param lat:latitude
        :param lon: longitude
        :param zoom: zoom level
        :return:
        """
        DEGREE = math.tau / 360
        lat = lat*DEGREE
        lon = lon*DEGREE
        mx = lon
        my = math.log(math.tan((lat + math.tau / 4) / 2))
        res = 2 ** (zoom + 8) / math.tau
        px = mx * res
        py = my * res
        return px, py

    def pixels2latlon(self, px, py, zoom):
        """
        Covert pixel locations to longitude and latitude
        original author: https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser/50536888#50536888
        :param lat:latitude
        :param lon: longitude
        :param zoom: zoom level
        :return: None
        """
        DEGREE = math.tau / 360
        res = 2 ** (zoom + 8) / math.tau
        mx = px / res
        my = py / res
        lon = mx
        lat = 2 * math.atan(math.exp(my)) - math.tau / 4
        return lat/DEGREE, lon/DEGREE


if __name__ == "__main__":
    reader = MapReader()
    reader.get_continuous_image(41.82518065811719, -71.39923535941084, 20,
                               "API_KEY", "images/download/continuous/", 2, 2)

    #for i in range(4):
    #    filename = str(i) + '.png'
    #    img = np.expand_dims(cv2.resize(cv2.imread("images/download/continuous/" + filename), (320, 320)), axis=0)/255.0
    #   print(reader.classify(img))
