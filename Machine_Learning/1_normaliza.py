import numpy as np
import cv2

def load_datasets(paths :list):
    print('Carregando')
    datasets = []
    for path in paths:
        dataset = np.load(path)
        datasets.append(dataset)
    return datasets

def normalize(datasets :list):
    print('Normalizando')
    n_datasets = []
    for dataset in datasets:
        n_images = []
        for image in dataset:
            n = image.astype(float)
            n = (n - 127.5) / 127.5
            n_images.append(n)
        n_datasets.append(n_images)
        print('Base Concluida')
    return n_datasets

def save_datasets(save_paths :list,datasets :list):
    print('Salvando Dados')
    for i in range(0, len(datasets)):
        np.save(save_paths[i], datasets[i])

# Carrega arquivo com imagens no formato npy
paths = ['npy_data/apple.npy','npy_data/axe.npy','npy_data/bicycle.npy','npy_data/book.npy','npy_data/bus.npy']
datasets = load_datasets(paths)

# Normaliza valor dos pixels entre -1 e 1
datasets = normalize(datasets)
print(datasets[0][0])

# Salva dados
save_paths = ['normalized_data/apple.npy','normalized_data/axe.npy','normalized_data/bicycle.npy','normalized_data/book.npy','normalized_data/bus.npy']
save_datasets(save_paths, datasets)
