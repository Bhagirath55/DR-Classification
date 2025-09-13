import cv2
import numpy as np


def preprocess_and_save_image(file, original_path, processed_path):
    """ Saves the uploaded image and applies preprocessing steps """

    # Save Original Image
    file.save(original_path)

    # Load Image (BGR Format)
    img = cv2.imread(original_path)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 224x224
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Convert to Grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE (ClipLimit = 3.0, TileGridSize = (8,8))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # Convert back to RGB
    img_processed = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    # Normalize the image (Scale Pixel Values Between 0 and 1)
    img_normalized = img_processed / 255.0

    # Save Preprocessed Image
    cv2.imwrite(processed_path, (img_processed * 255).astype(np.uint8))

    return img_normalized
