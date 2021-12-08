import os
import csv


def rename_files(not_city_dir, city_dir):
    """
    Rename the files in not city directories and city directories so they are in numerical order
    :param not_city_dir: Folder directory for not city images
    :param city_dir: Folder directory for city images
    :return: None
    """
    city_names = os.listdir(city_dir)
    not_city_names = os.listdir(not_city_dir)
    for i, name in enumerate(city_names):
        dst = f"../images/renamed_city/{str(i)}.png"
        src = os.path.join(city_dir, name)
        os.rename(src, dst)

    for i, name in enumerate(not_city_names):
        dst = f"../images/renamed_not_city/{str(i+len(city_names))}.png"
        src = os.path.join(not_city_dir, name)
        os.rename(src, dst)


def label_images(not_city_dir, city_dir, output_name):
    """
    Label images in city data as 1 and those in not city data as 0
    :param not_city_dir: Directory for not city images
    :param city_dir: Directory for city images
    :param output_name: Output csv file directory
    :return: None
    """
    city_names = os.listdir(city_dir)
    not_city_names = os.listdir(not_city_dir)
    labels = []
    for name in city_names:
        labels.append([name, 1])
    for name in not_city_names:
        labels.append([name, 0])
    with open(output_name, 'w', newline="\n") as f:
        csvwriter = csv.writer(f)
        for row in labels:
            csvwriter.writerow(row)


if __name__ == "__main__":
    rename_files("../images/not_city", "../images/city")
    label_images("../images/not_city", "../images/city", "images/label.csv")