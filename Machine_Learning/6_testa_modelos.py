import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import keras
import pickle

# Load models
model = keras.models.load_model('saved_models/main_model.h5')
file = open('saved_models/naive_model.pkl','rb')
model_naive = pickle.load(file)

# Carrega fotos de teste
img_app = np.load('cam_images/apple.npy')
img_axe = np.load('cam_images/axe.npy')
img_bic = np.load('cam_images/bicycle.npy')
img_boo = np.load('cam_images/book.npy')
img_bus = np.load('cam_images/bus.npy')

images = [img_app, img_axe, img_bic, img_boo, img_bus]
images = np.array(images)

# Faz Predição - CNN
labels = ['apple', 'axe', 'bicycle', 'book', 'bus']
prediction = model.predict_classes(images)
for pred in prediction:
    print(pred)

print('\n')

# Faz Predição - Naive
images = np.reshape(images, [-1, 784])
prediction = model_naive.predict(images)
for pred in prediction:
    print(pred)
