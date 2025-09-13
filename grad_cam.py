import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from utils import load_models

# Load models
googlenet, resnet, _, _, _, _, _, _, custom_cnn = load_models()

GRAD_CAM_FOLDER = "grad_cam"
os.makedirs(GRAD_CAM_FOLDER, exist_ok=True)


def get_latest_file(directory):
    """ Gets the most recently uploaded & processed image """
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ✅ Custom CNN Grad-CAM Function (With Error Handling)
def generate_grad_cam_custom_cnn(image_path, layer_name="conv2d_4"):
    """ Generates Grad-CAM for Custom CNN """

    try:
        # Load Image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"❌ Error: Could not load image {image_path}")
        print("✅ Image successfully loaded.")

        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize

        # Check Layer Name
        layer_names = [layer.name for layer in custom_cnn.layers]
        if layer_name not in layer_names:
            print(f"❌ Error: Layer '{layer_name}' not found in Custom CNN.")
            print(f"✅ Available layers: {layer_names}")
            return None

        # Create Grad-CAM Model
        grad_model = Model(inputs=custom_cnn.input,
                           outputs=[custom_cnn.get_layer(layer_name).output, custom_cnn.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, np.argmax(predictions)]

        # Compute Gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print(f"❌ No gradients found for layer {layer_name}.")
            return None
        print("✅ Gradients computed successfully.")

        # Apply Global Average Pooling to Gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0].numpy() * pooled_grads.numpy()
        heatmap = np.mean(conv_outputs, axis=-1)

        # Debug heatmap before normalization
        print(f"Heatmap shape before normalization: {heatmap.shape}")
        print(f"Heatmap min: {np.min(heatmap)}, max: {np.max(heatmap)}")

        # Normalize Heatmap
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        else:
            heatmap = np.zeros_like(heatmap)  # Prevent division by zero

        # Fix dtype issues
        heatmap = np.nan_to_num(heatmap)  # Replace NaN with 0
        heatmap = np.uint8(255 * heatmap)  # Convert to valid OpenCV format

        print(f"Heatmap shape after processing: {heatmap.shape}")

        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (224, 224))

        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # Save Grad-CAM Image
        cnn_grad_cam_path = os.path.join(GRAD_CAM_FOLDER, "cnn_grad_cam.jpg")
        cv2.imwrite(cnn_grad_cam_path, superimposed_img)
        print(f"✅ Saved Grad-CAM for Custom CNN to {cnn_grad_cam_path}")

        return cnn_grad_cam_path

    except Exception as e:
        print(f"❌ Error generating Grad-CAM for Custom CNN: {str(e)}")
        return None


# ✅ GoogleNet-SVM & ResNet-SVM Grad-CAM Function
def generate_grad_cam_svm(model, image_path, layer_name, output_filename):
    """ Generates Grad-CAM for GoogleNet & ResNet at the specified layer """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Could not load image {image_path}")
            return None

        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0

        # Create Grad-CAM Model with the correct layer
        grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, np.argmax(predictions)]

        # Compute Gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print(f"❌ Error: No gradients found for layer {layer_name}.")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs.numpy()[0] * pooled_grads.numpy()
        heatmap = np.mean(conv_outputs, axis=-1)

        # Normalize Heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else heatmap

        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (224, 224))

        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # Save Grad-CAM Image
        grad_cam_path = os.path.join(GRAD_CAM_FOLDER, output_filename)
        cv2.imwrite(grad_cam_path, superimposed_img)
        print(f"✅ Saved Grad-CAM to {grad_cam_path}")
        return grad_cam_path

    except Exception as e:
        print(f"❌ Error generating Grad-CAM for {output_filename}: {str(e)}")
        return None


# ✅ Function to Generate All Grad-CAMs
def generate_all_grad_cams():
    processed_folder = "processed"
    latest_image_path = get_latest_file(processed_folder)

    if not latest_image_path:
        return {"error": "No processed image found"}

    grad_cams = {}

    # Custom CNN Grad-CAM (if it fails, return error message)
    cnn_grad_cam_path = generate_grad_cam_custom_cnn(latest_image_path)
    grad_cams["grad_cam_cnn"] = f"/grad_cam/cnn_grad_cam.jpg" if cnn_grad_cam_path else ("Due to some issue, "
                                                                                         "we are unable to generate "
                                                                                         "the Grad-CAM for Custom CNN.")

    # GoogleNet-SVM Grad-CAM
    googlenet_grad_cam_path = generate_grad_cam_svm(googlenet, latest_image_path, "mixed10", "googlenet_grad_cam.jpg")
    if googlenet_grad_cam_path:
        grad_cams["grad_cam_googlenet"] = f"/grad_cam/googlenet_grad_cam.jpg"

    # ResNet-SVM Grad-CAM
    resnet_grad_cam_path = generate_grad_cam_svm(resnet, latest_image_path, "conv5_block3_out", "resnet_grad_cam.jpg")
    if resnet_grad_cam_path:
        grad_cams["grad_cam_resnet"] = f"/grad_cam/resnet_grad_cam.jpg"
    print("✅ Returning Grad-CAM results:", grad_cams)  # Debugging
    return grad_cams
