import requests
import random


def generate_random(api_key, directory, number=100):
    """
    Download google map images from random locations approximately within the US and
    save them in given directory
    :param api_key: Google Map api key
    :param number: Number of images
    :param directory: Directory to save the images
    :return: None
    """
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    zoom = 20

    for i in range(number):
        x = random.uniform(35.60317091263109, 47.28273421414902)
        y = random.uniform(-79.36051194302924, -119.9250320404095)
        center = str(x) + "," + str(y)
        r = requests.get(url+"center=" + center + "&zoom=" + str(zoom)
                         + "&size=640x640" + "&maptype=satellite" + "&key="
                         + api_key)

        f = open(directory + str(i) + '.png', 'wb')
        f.write(r.content)
        f.close()


def generate_city(api_key, directory, number=100):
    """
    Download images from random locations near several major US cities
    :param api_key: Google map api key
    :param directory: Directory to save the images
    :param number: Number of images to download
    :return: None
    """
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    zoom = 20
    cities = [[42.40795768506819, -71.0609861267691],
              [40.74712879975991, -74.05905367579919],
              [38.94885595098216, -77.06592976043397],
              [34.089480887230074, -118.2602997195616],
              [33.450057929731194, -112.07651882512168]]
    for i in range(number):
        cities_i = random.randint(0, 4)
        y = random.uniform(-0.03, 0.03)
        x = random.uniform(-0.03, 0.03)

        center = str(cities[cities_i][0] + x) + "," + str(cities[cities_i][1] + y)
        r = requests.get(url+"center=" + center + "&zoom=" + str(zoom)
                         + "&size=640x640" + "&maptype=satellite" + "&key="
                         + api_key)

        f = open(directory + str(i) + '.png', 'wb')
        f.write(r.content)
        f.close()


if __name__ == "__main__":
    generate_random("API_KEY", '../images/download/random/', 10)
    generate_city("API_KEY", '../images/download/city/', 10)
