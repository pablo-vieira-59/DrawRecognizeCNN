import numpy as np
import cv2

# Carrega arquivo com imagens no formato npy
data_bic = np.load('npy_data/bicycle.npy')
data_app = np.load('npy_data/apple.npy')
data_bus = np.load('npy_data/bus.npy')
data_boo = np.load('npy_data/book.npy')
data_axe = np.load('npy_data/axe.npy')

data = [data_app, data_axe, data_boo, data_bic, data_bus]

# Normaliza valor dos pixels entre 0-1
for label in data:
    for image in label:
        image = image.astype(float)
        # for i in range(0,len(image)):
        #image[i] = image[i]/255.0
        image = image.reshape([28, 28])
        cv2.imshow('window', image)
        cv2.waitKey(0)
    print('Base Concluida')

# Salva dados
print('Salvando Dados')
np.save('npy_normalized_data/apple.npy', data_app)
np.save('npy_normalized_data/axe.npy', data_axe)
np.save('npy_normalized_data/book.npy', data_boo)
np.save('npy_normalized_data/bicycle.npy', data_bic)
np.save('npy_normalized_data/bus.npy', data_bus)
