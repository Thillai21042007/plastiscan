from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# Load pretrained model (MobileNet - lightweight)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def analyze_soil(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((224,224))
    image = np.array(image)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    label = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0][1]

    # Logic mapping
    if "sand" in label:
        moisture = 30
        npk = 20
        microplastic = 10
        advice = "Add organic matter to improve fertility."
    elif "clay" in label:
        moisture = 70
        npk = 50
        microplastic = 20
        advice = "Ensure proper drainage."
    else:
        moisture = 50
        npk = 40
        microplastic = 15
        advice = "Balanced soil. Maintain nutrients."

    return {
        "detected_type": label,
        "microplastic": microplastic,
        "npk": npk,
        "moisture": moisture,
        "advice": advice
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    result = analyze_soil(file.read())
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)