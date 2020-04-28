import pandas as pd
import numpy as np
import time
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def load_dataset():
    X = np.load('training_data/X_train.npy')
    Y = np.load('training_data/Y_train.npy')
    testX = np.load('training_data/X_test.npy')
    testY = np.load('training_data/Y_test.npy')

    return X, Y , testX, testY

def train(model , X, Y, n_batch):
    start = time.time()
    n_batches = len(X)//n_batch
    X = np.split(X, n_batches)
    Y = np.split(Y, n_batches)
    classes = [1,2,3,4,5]
    for i in range(n_batches):
        model.partial_fit(X[i], Y[i], classes=classes)
        print('Batch:%d/%d' % (n_batches, i+1))
    end = time.time()
    elapsed = end - start
    print('Tempo de Treino:%.2f segundos' % (elapsed))
    file = open('saved_models/naive_model.pkl', 'wb')
    pickle.dump(model, file)
    file.close()

def calc_metrics(testY, y_pred):
    # Calculando F1 Score
    score_f1 = f1_score(testY, y_pred, average='weighted')

    # Calculando Precision Score
    score_pre = precision_score(testY, y_pred, average='weighted')

    # Calculando Accuracy Score
    score_acc = accuracy_score(testY, y_pred)

    # Calculando Recall Score
    score_recall = recall_score(testY, y_pred, average='weighted')

    return [score_f1, score_pre, score_acc, score_recall]

def save_metrics(scores, pred_time):
    row = ['Naive Bayes']
    for score in scores:
        row.append(score)
    row.append(pred_time)
    row = np.array(row)
    row = np.reshape(row, (1,6))

    # Salvando Metricas
    data = pd.DataFrame(row, columns=['Model','F1 Score','Precision Score', 'Accuracy Score','Recall Score', 'Prediction Time'])
    data.to_csv('metrics/naive_scores.csv', index=False, sep=';')

# Carregando dataset
X, Y, testX, testY = load_dataset()

# Instanciando Modelo
model = GaussianNB()

# Treinando Modelo
train(model, X, Y, n_batch=1000)

# Fazendo Predições
print('Calculando Metricas ...')
start = time.time()
y_pred = model.predict(testX)
end = time.time()
elapsed = end - start
print('Tempo de Classificação :%.2f segundos' % (elapsed))

# Calcula Metricas
scores = calc_metrics(testY, y_pred)

# Salvando Metricas
save_metrics(scores, elapsed)
