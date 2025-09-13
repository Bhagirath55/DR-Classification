from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from preprocessing import preprocess_and_save_image
from prediction import predict_dr_stage
from grad_cam import generate_all_grad_cams

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
GRAD_CAM_FOLDER = "grad_cam"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(GRAD_CAM_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["GRAD_CAM_FOLDER"] = GRAD_CAM_FOLDER


# 🔹 Serve HTML Page
@app.route("/")
def home():
    return render_template("index.html")


# 🔹 Upload & Preprocess Image
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    original_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], filename)

    # Save & Preprocess Image
    preprocess_and_save_image(file, original_path, processed_path)

    return jsonify({
        "original_url": f"/uploads/{filename}",
        "preprocessed_url": f"/processed/{filename}"
    })


# 🔹 Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    result = predict_dr_stage(app.config["PROCESSED_FOLDER"])
    return jsonify(result)


@app.route("/generate_grad_cam", methods=["POST"])
def generate_grad_cam():
    grad_cams = generate_all_grad_cams()
    return jsonify(grad_cams)


@app.route("/grad_cam/<filename>")
def get_grad_cam_image(filename):
    filepath = os.path.join(GRAD_CAM_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    print(f"Serving Grad-CAM Image: {filename}")
    return send_from_directory(GRAD_CAM_FOLDER, filename)


# 🔹 Serve Images
@app.route("/uploads/<filename>")
def get_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/processed/<filename>")
def get_preprocessed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
