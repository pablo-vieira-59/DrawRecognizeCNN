import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np
import pandas as pd
import time

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, LeakyReLU, BatchNormalization

# Constroi Modelo CNN
def BuildModel():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(7, 7), activation='relu', input_shape=[28, 28, 1]))
    model.add(MaxPool2D(pool_size=2))
    #14x14x64

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    #7x7x128

    model.add(Flatten())
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.4))
    #256x1

    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.4))
    #256x1

    model.add(Dense(5, activation='softmax'))
    #5x1

    adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def CompleteFit(X,Y,X_v,Y_v, n_train :int):
    for i in range(0, n_train):
        # Instancia novo modelo
        model = BuildModel()

        # Começa Treino
        start = time.time()
        history = model.fit(x=X, y=Y, batch_size=1000, epochs=5, verbose=1, validation_data=(X_v,Y_v))
        elapsed = time.time() - start

        # Salva Historico
        dataframe = pd.DataFrame(history.history)
        dataframe.to_csv('training_data_%d_%.2f_seconds.csv' % (i, elapsed), index=None)

#def GridSearchFit():
    #n_filters = [[16,32,64,128],[32,64,128,256],[16,32,64,128],[16,32,64,128]]

# Carrega o Dataset
X = np.load('training_data/X_train.npy')
Y = np.load('training_data/Y_train.npy')
testX = np.load('training_data/X_test.npy')
testY = np.load('training_data/Y_test.npy')

# Ajusta o Dataset
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
Y = Y.reshape([-1, 5])
testY = testY.reshape([-1, 5])

# Inicia Treinamento
CompleteFit(X, Y, testX, testY, 30)
