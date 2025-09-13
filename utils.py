import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3, ResNet50


def load_models():
    """ Loads all models & scalers for prediction """

    googlenet = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    googlenet_svm = joblib.load("models/sgd_svm.pkl")
    gn_scaler = joblib.load("models/scaler.pkl")
    gn_pca = joblib.load("models/pca.pkl")

    resnet_svm = joblib.load("models/sgd_svm_resnet16.pkl")
    res_scaler = joblib.load("models/scaler_resnet16.pkl")
    res_pca = joblib.load("models/pca_resnet16.pkl")

    custom_cnn = load_model("models/DR_CNN_model.h5")

    return googlenet, resnet, googlenet_svm, gn_scaler, gn_pca, resnet_svm, res_scaler, res_pca, custom_cnn
