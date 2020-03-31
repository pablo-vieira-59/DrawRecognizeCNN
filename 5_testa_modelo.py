import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import keras

# Load model
model = keras.models.load_model('saved_models/model_1.mdl')

# Carrega fotos de teste
img_app = np.load('cam_images/apple.npy')
img_axe = np.load('cam_images/axe.npy')
img_bic = np.load('cam_images/bicycle.npy')
img_boo = np.load('cam_images/book.npy')
img_bus = np.load('cam_images/bus.npy')

images = [img_app, img_axe, img_bic, img_boo, img_bus]

# Tranforma para shape de input
for img in images:
    img = img.reshape([28, 28])
images = np.asarray(images)
images = images.reshape([-1, 28, 28, 1])

# Faz Predição
labels = ['apple', 'axe', 'bicycle', 'book', 'bus']
prediction = model.predict(images)
for pred in prediction:
    print(labels[np.argmax(pred)])
    # print(pred)
