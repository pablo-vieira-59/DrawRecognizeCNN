import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB

def load_dataset():
    X = np.load('training_data/X_train.npy')
    Y = np.load('training_data/Y_train.npy')
    testX = np.load('training_data/X_test.npy')
    testY = np.load('training_data/Y_test.npy')

    return X, Y , testX, testY

# Carregando dataset
X, Y, testX, testY = load_dataset()

# Instanciando Modelo
model = GaussianNB()

# Treinando Modelo
n_batch = 1000
n_batches = len(X)//n_batch
X = np.split(X, n_batches)
Y = np.split(Y, n_batches)
classes = [1,2,3,4,5]
for i in range(n_batches):
    model.partial_fit(X[i], Y[i], classes=classes)
    print('Batch:%d/%d' % (n_batches, i+1))

# Calculando Acuracia com dados de Validação
score = model.score(testX, testY)
print(score)
