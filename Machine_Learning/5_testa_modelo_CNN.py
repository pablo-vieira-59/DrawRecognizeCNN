import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import keras

# Load model
model = keras.models.load_model('saved_models/main_model.h5')

# Carrega fotos de teste
img_app = np.load('cam_images/apple.npy')
img_axe = np.load('cam_images/axe.npy')
img_bic = np.load('cam_images/bicycle.npy')
img_boo = np.load('cam_images/book.npy')
img_bus = np.load('cam_images/bus.npy')

images = [img_app, img_axe, img_bic, img_boo, img_bus]
images = np.array(images)

# Faz Predição
labels = ['apple', 'axe', 'bicycle', 'book', 'bus']
#prediction = model.predict_classes(images)
prediction = model.predict(images) * 100
for pred in prediction:
    #print(labels[np.argmax(pred)])
    print(pred)
