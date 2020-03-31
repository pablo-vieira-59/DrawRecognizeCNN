import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import layers
from keras import optimizers
import tensorflow as tf
import numpy as np
import keras

def BuildModel():
    adam = optimizers.Adam(learning_rate=0.0001)

    layer_in = layers.Input(shape=(28, 28, 1), name='input')
    y = layers.Conv2D(filters=64, kernel_size=(7, 7), activation='relu')(layer_in)
    y = layers.MaxPool2D(pool_size=(2, 2))(y)
    #14x14x64
    y = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(y)
    y = layers.MaxPool2D(pool_size=(2, 2))(y)
    #7x7x128
    y = layers.Flatten()(y)
    y = layers.Dense(256, activation='tanh')(y)
    y = layers.Dropout(0.4)(y)
    #256x1
    y = layers.Dense(256, activation='tanh')(y)
    y = layers.Dropout(0.2)(y)
    #256x1
    layer_out = layers.Dense(5, activation='softmax')(y)
    #5x1

    model = keras.Model(inputs=layer_in, outputs=layer_out)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    return model

# Loading dataset
X = np.load('training_data/X_train.npy')
Y = np.load('training_data/Y_train.npy')
testX = np.load('training_data/X_test.npy')
testY = np.load('training_data/Y_test.npy')

# Reshaping data
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
Y = Y.reshape([-1, 5])
testY = testY.reshape([-1, 5])

# Building Model
model = BuildModel()

# Training Model
model.fit(x=X, y=Y, batch_size=1000, epochs=10, verbose=1, validation_data=(testX, testY))
model.save('saved_models/model_1.mdl')
