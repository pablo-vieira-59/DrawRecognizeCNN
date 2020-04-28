import cv2
import numpy as np

def process_image(img):
    # Converte para Grayscale
    n = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensiona imagem
    n = cv2.resize(n, (640, 480))

    # Rect Value
    frame_w = 640
    frame_h = 480
    box_size = 200

    fw_size = (frame_w//2)
    fh_size = (frame_h//2)
    b_size = box_size//2

    start_x = fw_size - b_size
    start_y = fh_size - b_size
    end_x = fw_size + b_size
    end_y = fh_size + b_size

    n = n[start_y:end_y, start_x:end_x]

    # Suavizando Imagem para tirar ruidos
    n = cv2.GaussianBlur(n, (3, 3), 0)

    # Binariza Imagem usando um Treshold Adaptavel
    n = cv2.adaptiveThreshold(n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

    # Suavizando Imagem para aumentar espessura do traço
    n = cv2.GaussianBlur(n, (11, 11), 0)

    # Binariza Imagem
    n = cv2.adaptiveThreshold(n, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)

    # Inverte cor
    n = ~n

    return n

def normalize_image(img):
    # Normaliza Imagem
    n = (img-127.5) / 127.5
    return n

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = process_image(frame)
    cv2.imshow('window', img)
    cv2.waitKey(15)

cap.release()
cv2.destroyAllWindows()