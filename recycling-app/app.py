"""
app.py
-------
Servidor Flask que:
1. Serve a página web (templates/index.html) com acesso à câmera do celular.
2. Recebe uma imagem (base64) via POST /predict e retorna a classe de
   material reciclável prevista pelo modelo treinado (train_model.py).

COMO RODAR
-----------
    pip install -r requirements.txt
    python app.py

Depois acesse pelo navegador do celular (mesma rede Wi-Fi do computador):
    http://<IP-DO-SEU-COMPUTADOR>:5000

IMPORTANTE SOBRE CÂMERA EM CELULAR
------------------------------------
Navegadores modernos só liberam acesso à câmera (getUserMedia) em:
  - localhost (funciona só no próprio computador)
  - conexões HTTPS

Para testar no celular pela rede local, você tem duas opções fáceis:
  1. Usar o ngrok (https://ngrok.com) para gerar uma URL https temporária:
         ngrok http 5000
     e acessar a URL https gerada pelo celular.
  2. Gerar um certificado autoassinado e rodar com:
         app.run(host="0.0.0.0", port=5000, ssl_context="adhoc")
     (aceite o aviso de "site não seguro" no celular)
"""

import base64
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "model/recycle_model.h5"
CLASS_INDEX_PATH = "model/class_indices.json"
IMG_SIZE = (224, 224)

model = None
class_names = None

# Dica de descarte por categoria (ajuste conforme a coleta seletiva da sua cidade)
DISPOSAL_TIPS = {
    "cardboard": "Papelão: dobre as caixas e mantenha secas antes de descartar.",
    "glass": "Vidro: embale cacos com jornal para não machucar os coletores.",
    "metal": "Metal: latas de alumínio/aço vão para a coleta de metais.",
    "paper": "Papel: evite papel sujo de gordura/comida, ele não é reciclável.",
    "plastic": "Plástico: enxágue embalagens antes de descartar.",
    "trash": "Rejeito: não reciclável, deve ir para o lixo comum.",
}


def load_model_and_classes():
    global model, class_names
    import tensorflow as tf  # import tardio para o servidor subir rápido

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado em '{MODEL_PATH}'. Rode train_model.py "
            "primeiro para treinar e salvar o modelo."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    # json guarda chaves como string; converte de volta pra int
    class_names = {int(k): v for k, v in idx_to_class.items()}


def preprocess_image(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo ainda não carregado no servidor."}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Nenhuma imagem recebida."}), 400

    try:
        # a imagem chega como data URL: "data:image/jpeg;base64,....."
        header, encoded = data["image"].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        return jsonify({"error": f"Falha ao decodificar imagem: {exc}"}), 400

    x = preprocess_image(image)
    preds = model.predict(x)[0]

    top_idx = int(np.argmax(preds))
    label = class_names[top_idx]
    confidence = float(preds[top_idx])

    ranking = sorted(
        [{"label": class_names[i], "confidence": float(p)} for i, p in enumerate(preds)],
        key=lambda d: d["confidence"],
        reverse=True,
    )

    return jsonify({
        "label": label,
        "confidence": confidence,
        "tip": DISPOSAL_TIPS.get(label, ""),
        "ranking": ranking,
    })


if __name__ == "__main__":
    load_model_and_classes()
    # host 0.0.0.0 para ser acessível por outros dispositivos na rede local
    app.run(host="0.0.0.0", port=5000, debug=True)
