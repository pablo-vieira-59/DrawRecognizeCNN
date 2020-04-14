import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# Carrega arquivo com imagens no formato npy
data_bic = np.load('normalized_data/bicycle.npy')
data_app = np.load('normalized_data/apple.npy')
data_bus = np.load('normalized_data/bus.npy')
data_boo = np.load('normalized_data/book.npy')
data_axe = np.load('normalized_data/axe.npy')

# Seleciona a base com menor quantidade
sizes = [len(data_app), len(data_axe), len(
    data_bic), len(data_boo), len(data_bus)]
m_value = sizes[np.argmin(sizes)]
print('Menor quantidade de samples : ', m_value)

# Ajusta bases para mesma quantidade
data_app = data_app[0:m_value-1, :]
data_axe = data_axe[0:m_value-1, :]
data_bic = data_bic[0:m_value-1, :]
data_boo = data_boo[0:m_value-1, :]
data_bus = data_bus[0:m_value-1, :]

sizes = [len(data_app), len(data_axe), len(
    data_bic), len(data_boo), len(data_bus)]
print('Tamanho ajustado : ', sizes, '\n')

# Adiciona labels
labels = []
for image in data_app:
    labels.append(1)

for image in data_axe:
    labels.append(2)

for image in data_bic:
    labels.append(3)

for image in data_boo:
    labels.append(4)

for image in data_bus:
    labels.append(5)

# Junta imagens das classes em um array só
data = np.concatenate((data_app, data_axe, data_bic, data_boo, data_bus))

# One Hot Encoding
encoded = pd.get_dummies(labels)

# Cria Dataset
data = pd.DataFrame(data=data)
data['apple'] = encoded.iloc[:, 0]
data['axe'] = encoded.iloc[:, 1]
data['bicycle'] = encoded.iloc[:, 2]
data['book'] = encoded.iloc[:, 3]
data['bus'] = encoded.iloc[:, 4]
data['classes'] = labels
X = data.iloc[:, :784]
Y = data.iloc[:, 784:789]

print('Quantidade por classe :', '\n', data['classes'].value_counts(), '\n')

# Divide dados de teste e treino , estratificado
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0, stratify=data['classes'])

# Salva Arquivos
print('Salvando dados')
np.save('training_data/X_train.npy', X_train)
np.save('training_data/X_test.npy', X_test)
np.save('training_data/Y_train.npy', Y_train)
np.save('training_data/Y_test.npy', Y_test)
