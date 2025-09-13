import os
import cv2
import numpy as np
from utils import load_models

# Load models
googlenet, resnet, googlenet_svm, gn_scaler, gn_pca, resnet_svm, res_scaler, res_pca, custom_cnn = load_models()

# DR Stage Mapping
dr_stages = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


def get_latest_file(directory):
    """ Gets the most recently uploaded & processed image """
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f)]  # Ensure only files
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)  # Sort by modification time
    return latest_file


def extract_features(img_array, model):
    """ Extracts features using a given CNN model (GoogleNet or ResNet) """
    features = model.predict(img_array)
    return features.reshape(1, -1)


def clamp_confidence(value):
    return max(1, min(100, round(value*100)))


def predict_dr_stage(processed_folder):
    """ Predicts DR stage using GoogleNet+SVM, ResNet+SVM, and Custom CNN """

    # Get the most recently uploaded image
    image_path = get_latest_file(processed_folder)
    if not image_path:
        return {"error": "No processed image found"}

    # Load Preprocessed Image
    img = cv2.imread(image_path)

    # Normalize Image for Model Input
    img_normalized = img / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # 🔹 Custom CNN Prediction
    cnn_pred = custom_cnn.predict(img_expanded)
    cnn_label = np.argmax(cnn_pred)
    cnn_confidence = clamp_confidence(np.max(cnn_pred))

    # 🔹 GoogleNet-SVM Prediction
    gn_features = extract_features(img_expanded, googlenet)
    gn_features_scaled = gn_pca.transform(gn_scaler.transform(gn_features))
    gn_label = googlenet_svm.predict(gn_features_scaled)[0]
    gn_confidence = clamp_confidence(max(googlenet_svm.decision_function(gn_features_scaled)[0]))

    # 🔹 ResNet-SVM Prediction
    res_features = extract_features(img_expanded, resnet)
    res_features_scaled = res_pca.transform(res_scaler.transform(res_features))
    res_label = resnet_svm.predict(res_features_scaled)[0]
    res_confidence = clamp_confidence(max(resnet_svm.decision_function(res_features_scaled)[0]))

    return {
        "cnn_prediction": {"stage": dr_stages[cnn_label], "confidence": cnn_confidence},
        "googlenet_svm_prediction": {"stage": dr_stages[gn_label], "confidence": gn_confidence},
        "resnet_svm_prediction": {"stage": dr_stages[res_label], "confidence": res_confidence}
    }
