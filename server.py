from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

def analyze_soil(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    avg_color = np.mean(image, axis=(0,1))
    brightness = np.mean(avg_color)

    # Moisture estimation
    moisture = int(100 - (brightness % 100))

    # NPK estimation (green dominance)
    npk = int((avg_color[1] / 255) * 100)

    # Microplastic estimation (texture noise)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    variance = np.var(gray)
    microplastic = int(variance % 100)

    return {
        "microplastic": microplastic,
        "npk": npk,
        "moisture": moisture,
        "advice": "Use organic compost and reduce plastic contamination."
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    result = analyze_soil(file.read())
    return jsonify(result)

# IMPORTANT FOR RENDER
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
 
