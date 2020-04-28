import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

def get_sizes(datasets :list):
    sizes = []
    for dataset in datasets:
        size = len(dataset)
        sizes.append(size)
    return sizes

def resize_datasets(datasets :list, max_val :int):
    rez_datasets = []
    for dataset in datasets:
        d = dataset[0:max_val, :]
        rez_datasets.append(d)
    return rez_datasets

def load_datasets(paths :list):
    datasets = []
    for path in paths:
        dataset = np.load(path)
        datasets.append(dataset)
    return datasets

def generate_labels(datasets :list):
    # Adiciona labels
    classes = []
    for i in range(0,len(datasets)):
        labels = np.ones(len(datasets[i]))
        labels = labels + i
        classes = np.concatenate([classes, labels])
    return classes

def prepare_dataset(datasets :list):
    # Calcula Tamanho dos datasets
    sizes = get_sizes(datasets)
    
    # Seleciona a base com menor quantidade
    max_value = sizes[np.argmin(sizes)]
    print('Menor quantidade de samples : ', max_value)

    # Ajusta valor max para multiplo de n_batch
    max_value = 100000

    # Ajusta bases para mesma quantidade
    rez_datasets = resize_datasets(datasets, max_value)

    sizes = get_sizes(rez_datasets)
    print('Tamanho ajustado : ', sizes, '\n')

    # Gera Labels
    labels = generate_labels(rez_datasets)

    # Junta imagens das classes em um array só
    data = np.concatenate(rez_datasets)

    return data, labels

def generate_train_data(hot_encoding :bool, labels :list, data :np.array):
    # One Hot Encoding
    encoded = pd.get_dummies(labels)
    data = pd.DataFrame(data=data)
    data['apple'] = encoded.iloc[:, 0]
    data['axe'] = encoded.iloc[:, 1]
    data['bicycle'] = encoded.iloc[:, 2]
    data['book'] = encoded.iloc[:, 3]
    data['bus'] = encoded.iloc[:, 4]
    data['classes'] = labels
    X = data.iloc[:, :784]
    Y = []
    if hot_encoding:
        Y = data.iloc[:, 784:789]
    else:
        Y = data['classes']

    return X, Y

# Carrega arquivo com imagens no formato npy
paths = ['npy_data/apple.npy','npy_data/axe.npy','npy_data/bicycle.npy','npy_data/book.npy','npy_data/bus.npy']
paths_n = ['normalized_data/apple.npy','normalized_data/axe.npy','normalized_data/bicycle.npy','normalized_data/book.npy','normalized_data/bus.npy']
datasets = load_datasets(paths_n)

for dataset in datasets:
    print(len(dataset))

# Reajusta dataset
data, labels = prepare_dataset(datasets)

# Gera dados de treino
X, Y = generate_train_data(hot_encoding=True, labels=labels, data=data)
print('Formato X:', X.shape)
print('Formato Y:', Y.shape)

# Divide dados de teste e treino , estratificado
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)

# Salva Arquivos
print('Salvando dados')
np.save('training_data/X_train.npy', X_train)
np.save('training_data/X_test.npy', X_test)
np.save('training_data/Y_train.npy', Y_train)
np.save('training_data/Y_test.npy', Y_test)
