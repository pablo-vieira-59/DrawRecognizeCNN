import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import urllib.request as ur
import json , requests, keras , cv2
from flask import Flask, make_response, render_template, Request, jsonify, Response
import tensorflow as tf

# Set CPU as available physical device
tf.config.set_visible_devices([], 'GPU')

# Loading Model
model = tf.keras.models.load_model('../Machine_Learning/saved_models/main_model.h5')

# Rect Value
start_x = 400 - 150
start_y = 300 - 150
end_x = 400 + 150
end_y = 300 + 150

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

def load_camera():
    url = "http://10.0.0.12:5959/photo.jpg"
    image = ur.urlopen(url,timeout=2)
    image = np.array(bytearray(image.read()),dtype=np.uint8)
    image = cv2.imdecode(image,-1)
    return image

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

@app.route("/data",methods=["GET","POST"])
def make_prediction():
    global model
    img = load_camera()
    x_pred = process_image(img)
    pred = model.predict_classes(x_pred)
    return jsonify({'class':int(pred[0])})
        


if __name__ == "__main__":  
      
    app.run(debug=True,port=80,host="0.0.0.0")
    