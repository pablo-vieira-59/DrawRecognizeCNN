import cv2
import numpy as np

# Carrega Imagens
labels = ['apple', 'axe', 'bicycle', 'book', 'bus']

img_app = cv2.imread('cam_images/t1.png')
img_axe = cv2.imread('cam_images/t2.png')
img_bic = cv2.imread('cam_images/t3.png')
img_boo = cv2.imread('cam_images/t4.png')
img_bus = cv2.imread('cam_images/t5.png')

images = [img_app, img_axe, img_bic, img_boo, img_bus]

i = 0
for img in images:
    # Converte para Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensiona imagem
    img = cv2.resize(img, (256, 256))

    # Suavizando Imagem para tirar ruidos
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Binariza Imagem usando um Treshold Adaptavel
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

    # Suavizando Imagem para aumentar espessura do traço
    img = cv2.GaussianBlur(img, (11, 11), 0)

    # Binariza Imagem
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

    # Inverte cor
    img = ~img

    # Mostra Imagem
    cv2.imshow('window', img)
    cv2.waitKey()

    # Redimensiona imagem
    img = cv2.resize(img, (28, 28))

    # Salva Imagem
    np.save('cam_images/' + labels[i] + '.npy', img)
    i += 1
