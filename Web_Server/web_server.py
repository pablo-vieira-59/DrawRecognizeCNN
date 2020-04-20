import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import urllib.request as ur
import json , requests, keras , cv2
from flask import Flask, make_response, render_template, Request, jsonify


app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def main():
    return render_template("index.html")

@app.route("/components",methods=["GET","POST"])
def components():
    return render_template("components.html")

@app.route("/data",methods=["GET","POST"])
def services_status():
    img = load_camera()
    img = process_image(img)
    label = make_prediction(img)
    return jsonify({'class':label})
    
def get_status(uri :str):
    try:
        response = requests.get(uri)
        response = response.text
        response = json.loads(response)
        status = response['status']
        return status
    except:
        return False

def load_camera():
    image = ur.urlopen(url,timeout=2)
    image = np.array(bytearray(image.read()),dtype=np.uint8)
    image = cv2.imdecode(image,-1)
    return image

def process_image(img):
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

    return None

def make_prediction(img):
    pred = model.predict_classes(img)
    labels = np.argsort(pred)[::-1]
    return labels[0]


if __name__ == "__main__":
    url = ""
    model = keras.models.load_model('../saved_models/model_1.mdl')
    app.run(debug=True,port=80,host="0.0.0.0")
    