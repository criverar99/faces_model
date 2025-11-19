import json
import os
import numpy as np
import cv2
import requests

# Endpoint de tu modelo
SERVER_URL = "https://criverar99-tensorflow-faces.onrender.com/v1/models/faces-model:predict"
IMAGE_PATH = f'{os.getcwd()}/data/carlos/number.jpeg'



def predict(photoNumber: int):
    # Cargar imagen en escala de grises
    img = cv2.imread(IMAGE_PATH.replace('number', str(photoNumber)), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))

    # Normalizar y adaptar dimensiones
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (150,150,1)
    img = np.expand_dims(img, axis=0)   # (1,150,150,1)

    # JSON para TensorFlow Serving
    predict_request = json.dumps({"instances": img.tolist()})
    headers = {"content-type": "application/json"}

    # POST al servidor
    response = requests.post(SERVER_URL, data=predict_request, headers=headers)
    response.raise_for_status()

    # Procesar predicción
    prediction = response.json()["predictions"][0]
    clases = ["adrian", "adsoft", "jessi", "simon", "carlos"]
    print("Predicciones (probabilidades):", prediction)
    print("Predicción final:", clases[np.argmax(prediction)])

def main():
    for i in range(300):
        print(f"Predicción número {i+1}:")
        predict(i+1)

if __name__ == "__main__":
    main()