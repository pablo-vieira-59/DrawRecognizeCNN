import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np
import pandas as pd
import time

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, LeakyReLU, BatchNormalization
from keras.models import load_model

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def reverse_one_hot(y_pred):
    n_pred = []
    for pred in y_pred:
        n_class = np.argmax(pred)
        n_pred.append(n_class)
    return n_pred

def build_model():
    model = Sequential()

    #28x28x1
    model.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu', input_shape=[28, 28, 1]))
    model.add(MaxPool2D(pool_size=2))
    #14x14x64

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
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

    adam = optimizers.Adam(learning_rate=0.00002)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def build_model(param :list):
    model = Sequential()

    #28x28x1
    model.add(Conv2D(filters=param[0], kernel_size=(4, 4), activation='relu', input_shape=[28, 28, 1]))
    model.add(MaxPool2D(pool_size=2))
    #14x14x64

    model.add(Conv2D(filters=param[1], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    #7x7x128

    model.add(Flatten())
    model.add(Dense(param[2], activation='tanh'))
    model.add(Dropout(0.4))
    #256x1

    model.add(Dense(param[3], activation='tanh'))
    model.add(Dropout(0.4))
    #256x1

    model.add(Dense(5, activation='softmax'))
    #5x1

    adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def multiple_models_fit(X, Y, x_test, y_test, n_train :int):
    for i in range(0, n_train):
        # Instancia novo modelo
        model = build_model()

        # Começa Treino
        start = time.time()
        history = model.fit(x=X, y=Y, batch_size=1000, epochs=5, verbose=1, validation_data=(x_test, y_test))
        elapsed = time.time() - start

        # Salva Historico
        dataframe = pd.DataFrame(history.history)
        dataframe.to_csv('training_data_%d_%.2f_seconds.csv' % (i, elapsed), index=False)

def grid_search_fit(X, Y, x_test, y_test):
    layer_1 = [16,32,64,128]
    layer_2 = [32,64,128,256]
    layer_3 = [8,16,32,64]
    layer_4 = [8,16,32,64]
    for p in layer_1:
        for q in layer_2:
            for r in layer_3:
                for s in layer_4:
                    model = build_model([p,q,r,s])
                    model.fit(x=X, y=Y, batch_size=1000, epochs=3, verbose=1, validation_data=(x_test, y_test))

def default_fit(X, Y, x_test, y_test):
    # Constroi Modelo
    model = build_model()

    # Inicia Treinamento
    start = time.time()
    history = model.fit(X, Y, batch_size=1000, epochs=10, validation_data=(x_test, y_test))
    end = time.time()
    elapsed = end - start
    print('Tempo de Treino :%.2f segundos' % (elapsed))

    # Salva Modelo
    model.save('saved_models/main_model.h5')

    # Salva Historico
    dataframe = pd.DataFrame(history.history)
    dataframe.index.name = 'Epoch'
    dataframe.to_csv('metrics/final_train_history.csv', index=True)
    return model

def calc_metrics(y_pred, y_true):
    # Reverte One Hot Encoding
    y_true = reverse_one_hot(y_true)

    # Calculando F1 Score
    score_f1 = f1_score(y_true, y_pred, average='weighted')

    # Calculando Precision Score
    score_pre = precision_score(y_true, y_pred, average='weighted')

    # Calculando Accuracy Score
    score_acc = accuracy_score(y_true, y_pred)

    # Calculando Recall Score
    score_recall = recall_score(y_true, y_pred, average='weighted')

    return [score_f1, score_pre, score_acc, score_recall]

def save_metrics(scores, pred_time):
    row = ['CNN']
    for score in scores:
        row.append(score)
    row.append(pred_time)
    row = np.array(row)
    row = np.reshape(row, (1,6))

    # Salvando Metricas
    data = pd.DataFrame(row, columns=['Model','F1 Score','Precision Score', 'Accuracy Score','Recall Score', 'Prediction Time'])
    data.to_csv('metrics/cnn_scores.csv', index=False, sep=';')

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
#grid_search_fit(X, Y, testX, testY) # Faz Grid Search
#multiple_models_fit(X, Y, testX, testY, 30) # Treina Multiplos Modelos
model = default_fit(X, Y, testX, testY) # Treina Modelo Final

# Calcula Metricas
start = time.time()
y_pred = model.predict_classes(testX)
end = time.time()
elapsed = end - start
print('Tempo de Classificação :%.2f segundos' % (elapsed))

# Salvando Metricas
scores = calc_metrics(y_pred, testY)
save_metrics(scores, elapsed)
