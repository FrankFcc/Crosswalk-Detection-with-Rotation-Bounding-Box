import tensorflow as tf


# A Deep Convolution model that follows a structure similar to resnet but shallower
class Resnet(tf.keras.Model):
    def __init__(self, learning_rate, batch_size, input_size):
        super(Resnet, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.batch_size = batch_size
        self.loss_list = []
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=input_size))
        self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=7, strides=2, padding='same'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
        self.model.add(ResnetBlock(16, 3, 1))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same'))
        self.model.add(ResnetBlock(32, 3, 2))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same'))
        self.model.add(ResnetBlock(64, 3, 2))
        self.model.add(ResnetBlock(64, 3, 1))
        self.model.add(ResnetBlock(128, 3, 2))
        self.model.add(ResnetBlock(128, 3, 1))
        self.model.add(tf.keras.layers.Reshape((-1,)))
        self.model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='normal'))
        self.model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='normal'))
        self.model.add(tf.keras.layers.Dense(2, kernel_initializer='normal'))

    def call(self, input_tensor):
        return self.model(input_tensor)

    def loss_function(self, logits, labels):
        return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=1))

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# One residual network block with two convolutions
class ResnetBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ResnetBlock, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters, kernel_size, 1, padding='same')
        self.short_cut_1 = tf.keras.layers.Conv2D(filters, 1, strides, padding='same')

    def call(self, input_tensor):
        net_in = self.conv_1(input_tensor)
        net_act = tf.nn.relu(net_in)
        net_in = self.conv_2(net_act)
        net_act = tf.nn.relu(net_in)
        net_act += self.short_cut_1(input_tensor)
        return tf.nn.relu(net_act)


