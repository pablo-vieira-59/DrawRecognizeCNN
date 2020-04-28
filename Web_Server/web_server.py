import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import urllib.request as ur
import json , requests, keras , cv2
from flask import Flask, make_response, render_template, Request, jsonify, Response, request
import tensorflow as tf
import re
import base64

# Set CPU as available physical device
tf.config.set_visible_devices([], 'GPU')

# Loading Model
model = tf.keras.models.load_model('../Machine_Learning/saved_models/main_model.h5')

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

app = Flask(__name__)

def get_status(uri :str):
    try:
        response = requests.get(uri)
        response = response.text
        response = json.loads(response)
        status = response['status']
        return status
    except:
        return False

def draw():
    while True:
        img = load_camera()
        img = apply_frame(img)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img.tobytes() + b'\r\n\r\n')

def apply_frame(img):
    color = (0, 0, 255)
    
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, 2)
    retval , frame = cv2.imencode('.png',img)
    return frame

@app.route("/load_camera",methods=["GET","POST"])
def load_camera():
    data = request.values['image']
    img = re.sub('^data:image/.+;base64,', '', data)
    img = base64.b64decode(img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    return make_prediction(img)

def process_image(img):
    # Recorta Imagem
    img = img[start_y:end_y, start_x:end_x]

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
    #cv2.imshow('window', img)
    #cv2.waitKey()

    # Redimensiona imagem
    img = cv2.resize(img, (28, 28))

    # Normaliza Imagem
    img = (img - 127.5)/127.5

    # Ajusta para entrada de Rede Naural
    x_pred = np.array([img])
    x_pred = np.reshape(x_pred, [1,28,28,1])

    return x_pred

@app.route("/",methods=["GET","POST"])
def main():
    return render_template("index.html")

@app.route("/components",methods=["GET","POST"])
def components():
    return render_template("components.html")

@app.route("/framed",methods=["GET","POST"])
def display_frame():
    return Response(draw(), mimetype='multipart/x-mixed-replace; boundary=frame')

def make_prediction(img):
    global model
    x_pred = process_image(img)
    pred = model.predict_classes(x_pred)
    return jsonify({'class':int(pred[0])})
        


if __name__ == "__main__":  
    app.run(debug=True,port=8080,host="0.0.0.0")
    