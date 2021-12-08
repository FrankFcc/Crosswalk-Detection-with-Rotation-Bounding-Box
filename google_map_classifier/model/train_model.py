from model import Resnet
import tensorflow as tf
from utils.preprocess import get_data
import numpy as np
from matplotlib import pyplot as plt


def shuffle_data(x, y):
    """
    Shuffles given data and label
    :param x: Dataset
    :param y: Labels
    :return: Randomly shuffled data and labels
    """
    sample_indices = np.arange(0, len(y))
    np.random.shuffle(sample_indices)
    shuffled_x = np.take(x, sample_indices, axis=0)
    shuffled_y = np.take(y, sample_indices, axis=0)
    return shuffled_x, shuffled_y


def separate_train_test(x, y, ratio):
    """
    Separate a train set and a test set by the ratio provided
    :param x: Data
    :param y: Labels
    :param ratio: Percentage of data used as training data
    :return: Separated training and testing data and labels
    """
    train_size = int(x.shape[0] * ratio)
    train_indices = np.arange(stop=train_size)
    test_indices = np.arange(x.shape[0] - train_size) + train_size

    return np.take(x, train_indices, axis=0), np.take(y, train_indices, axis=0), \
           np.take(x, test_indices, axis=0), np.take(y, test_indices, axis=0)


def cutout(img, size):
    """
    Implementation of cutout from Terrance DeVries and Graham W. Taylor, 2017. https://arxiv.org/abs/1708.04552
    :param img: image to be augmented
    :param size: int size of cutout region
    """
    height = img.shape[0]
    width = img.shape[1]
    mask = np.ones(img.shape, np.float32)
    x = np.random.randint(width)
    y = np.random.randint(height)

    y_1 = int(max(y - size / 2.0, 0))
    y_2 = int(min(y + size / 2.0, height))
    x_1 = int(max(x - size / 2.0, 0))
    x_2 = int(min(x + size / 2.0, width))
    mask[x_1:x_2, y_1:y_2, :] = 0
    return img * mask


def train(model, inputs, labels):
    """
    Train the model for 1 epoch
    :param model: A resnet model
    :param inputs: Images of shape (batch,width, height, color channel)
    :param labels: One-hot coded labels
    :return: None
    """
    max_iter_per_epoch = max(int(inputs.shape[0] / model.batch_size), 1)
    for i in range(max_iter_per_epoch):
        end_point = (i + 1) * model.batch_size
        batch_input = inputs[i * model.batch_size:end_point, :, :, :]
        batch_labels = labels[i * model.batch_size:end_point]
        with tf.GradientTape() as tape:
            logits = model(tf.image.random_flip_left_right(batch_input))
            loss = model.loss_function(logits, batch_labels)
            model.loss_list.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if i % 20 == 0:
            train_acc = model.accuracy(logits, batch_labels)
            print("Batch Accuracy on training set is: " + str(train_acc.numpy()))
            print("Batch average loss is " + str(loss.numpy()))


def test(model, test_inputs, test_labels):
    """
    Compute test accuracy of model
    :param model: A resnet model
    :param test_inputs: Test data
    :param test_labels: Test labels
    :return: None
    """
    tail = test_inputs.shape[0] % model.batch_size
    max_iter_per_epoch = max(int(test_inputs.shape[0] / model.batch_size), 1)
    accuracies = []
    for i in range(max_iter_per_epoch):
        if i + 1 > max_iter_per_epoch:
            end_point = i * model.batch_size + tail
        else:
            end_point = (i + 1) * model.batch_size
        batch_input = test_inputs[i * model.batch_size:end_point, :]
        batch_labels = test_labels[i * model.batch_size:end_point, :]
        logits = model.call(batch_input)
        accuracies.append(model.accuracy(logits, batch_labels))
    return np.average(accuracies)


if __name__ == "__main__":
    images, labels = get_data("../images/all_images", "../images/label.csv")
    print(images.shape)
    print(labels.shape)
    images, labels = shuffle_data(images, labels)
    train_x, train_y, test_x, test_y = separate_train_test(images, labels, 0.9)
    model = Resnet(0.0001, 200, (320, 320, 3))
    for i in range(20):
        train(model, train_x, train_y)
        train_x, train_y = shuffle_data(train_x, train_y)
    test_accuracy = test(model, test_x, test_y)
    print("Test set accuracy is: " + str(test_accuracy))
    plt.plot(np.arange(len(model.loss_list)), model.loss_list)
    plt.show()
    # Save the network for inference in MapReader
    model.save("../saved_model/")
