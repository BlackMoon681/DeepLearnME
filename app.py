from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model import load_model, predict_image
import os
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)
model = load_model()  # chargé une seule fois au démarrage

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

analyses = []  # In-memory storage for history (no DB)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "service": "SkinDetect API (Flask)"})


@app.post("/predict")
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file:
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        pred_class, conf, status = predict_image(filepath, model)

        analysis = {
            "id": str(uuid.uuid4()),
            "prediction": pred_class,
            "confidence": float(conf),
            "status": status,
            "image_path": f"/uploads/{filename}",
            "created_at": datetime.utcnow().isoformat(),
        }
        analyses.append(analysis)
        return jsonify(analysis)


@app.get("/analysis/history")
def get_history():
    return jsonify(analyses)


@app.get('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)