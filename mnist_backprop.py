import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import time

batch_size = 128
num_classes = 10


train_begin_time = time.time()
best_score = 0

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def save_weights(model, filename, layer):
    conv1 = model.get_layer('conv{0}'.format(layer)).get_weights()
    fc1 = model.get_layer('fc1').get_weights()
    fc2 = model.get_layer('fc2').get_weights()

    np.savez(filename, W_conv=conv1[0], b_conv=conv1[1], W_fc1=fc1[0], b_fc1=fc1[1],
                W_fc2=fc2[0], b_fc2=fc2[1])

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.epoch_times = []
        self.best_weights = None

    def on_epoch_begin(self, epoch, logs={}):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs={}):
        global best_score
        self.times.append(time.time() - train_begin_time)
        self.epoch_times.append(time.time() - self.t0)

        if logs.get('val_acc') > best_score:
            try:
                best_score = logs.get('val_acc')
                self.best_weights = save_weights(self.model, 'weights_layer3.npz', 3)
            except Exception:
                pass

def run_backprop():
    main_input = Input(shape=input_shape, name='main_input')
    conv1 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv1')(main_input)
    conv2 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv2')(conv1)

    conv2 = MaxPooling2D(pool_size = (2,2))(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv3')(conv2)
    conv3_drop = Dropout(.3)(conv3)
    conv3_flat = Flatten()(conv3_drop)

    fc1 = Dense(150, activation='relu', name='fc1')(conv3_flat)
    fc1_drop = Dropout(.5)(fc1)
    main_output = Dense(10, activation='softmax', name='fc2')(fc1_drop)

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    time_history = TimeHistory()
    history = keras.callbacks.History()

    def schedule(epoch):
        if epoch < 2:
            return 0.005
        elif epoch < 10:
            return 0.002
        elif epoch < 40:
            return 0.001
        elif epoch < 60:
            return 0.0005
        elif epoch < 80:
            return 0.0001
        else:
            return 0.00005

    rate_schedule = keras.callbacks.LearningRateScheduler(schedule)

    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        rotation_range=7,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
        zoom_range=.1) 
    # Compute quantities required for feature-wise normalization
    datagen.fit(x_train)

    time_history = TimeHistory()
    history = keras.callbacks.History()

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=100, callbacks=[history, time_history, rate_schedule],
                        validation_data=(x_test, y_test))

    np.savez('mnist_backprop_results.npz', acc=history.history['acc'], loss=history.history['loss'],
              val_acc=history.history['val_acc'], val_loss=history.history['val_loss'],
              times=time_history.times, epoch_times=time_history.epoch_times)

if __name__ == "__main__":
    run_backprop()
