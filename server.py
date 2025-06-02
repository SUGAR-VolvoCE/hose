from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

# Load the trained model once
model = YOLO("runs/train/exp/weights/hose.pt")  # update path if needed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def readb64_image(base64_str):
    # Remove prefix
    base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        image_b64 = data["image"]
        print("jksndask")

        # Decode image from base64
        img = readb64_image(image_b64)

        # Run YOLO prediction
        results = model.predict(source=img, device=device, imgsz=640)

        # Get the first result and draw annotations
        annotated_img = results[0].plot()

        # Encode back to base64
        _, buffer = cv2.imencode(".jpg", annotated_img)
        result_b64 = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"processed_image": f"data:image/jpeg;base64,{result_b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9002)
