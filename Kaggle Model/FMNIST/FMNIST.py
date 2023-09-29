#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('./fmnist/fashion-mnist_train.csv')
test_data = pd.read_csv('./fmnist/fashion-mnist_test.csv')
print("Fashion MNIST train -  rows:",train_data.shape[0]," columns:", train_data.shape[1])
print("Fashion MNIST test -  rows:",test_data.shape[0]," columns:", test_data.shape[1])

# data preprocessing
def data_preprocessing(raw):
    out_y = keras.utils.np_utils.to_categorical(raw.label, 10)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python import keras
from keras.utils import np_utils

# prepare the data
X, y = data_preprocessing(train_data)
X_test, y_test = data_preprocessing(test_data)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)
print("training set size",x_train.shape[0], x_train.shape[1:4])
print("validation set size",x_val.shape[0], x_val.shape[1:4])
print("test set size",X_test.shape[0]," columns:", X_test.shape[1:4])

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
# Model
#cnn = Sequential()
#cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#cnn.add(BatchNormalization())
#
#cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
#cnn.add(BatchNormalization())
#cnn.add(Dropout(0.25))
#
#cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#cnn.add(BatchNormalization())
#cnn.add(Dropout(0.25))
#
#cnn.add(Flatten())
#cnn.add(Dense(512, activation='relu'))
#cnn.add(BatchNormalization())
#cnn.add(Dropout(0.5))
#
#cnn.add(Dense(128, activation='relu'))
#cnn.add(BatchNormalization())
#cnn.add(Dropout(0.5))
#
#cnn.add(Dense(10, activation='softmax'))

model = Sequential()

model.add(Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_normal', input_shape=(28,28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
train_model = model.fit(x_train, y_train,
                  batch_size=256,
                  epochs=100,
                  validation_data=(x_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])