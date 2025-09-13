# 👁️ Early Detection of Diabetic Retinopathy (DR)  
*Deep Learning for Smarter Healthcare*  

---

## 📌 Overview  
Diabetic Retinopathy (DR) is one of the leading causes of blindness worldwide. Early detection can save vision — but manual screening is slow, error-prone, and resource-intensive.  

This project leverages **Deep Learning** to automatically detect and classify different stages of DR from retinal images. By combining **pretrained models (GoogLeNet, ResNet)**, a **custom CNN**, and **Grad-CAM for interpretability**, the system offers reliable predictions and transparent decision-making.  

---

## 🚀 Features  
- 🔍 **Multi-Model Approach**: Pretrained feature extractors (GoogLeNet, ResNet) + Custom CNN.  
- 🎯 **Stage Classification**: No DR → Proliferative DR (5 stages).  
- 🖼️ **Preprocessing Pipeline**: CLAHE, gamma correction, sharpening, normalization.  
- 🌈 **Explainability**: Grad-CAM heatmaps to visualize what the model “sees.”  
- 🖥️ **User Interface**: Upload images → preprocessing → prediction → Grad-CAM visualization.  

---

## 📂 Dataset  
- **Source**: Kaggle (EyePACS, APTOS, Messidor).  
- **Size**: 92,501 retinal images.  
- **Labels**:  
  - 0 → No DR  
  - 1 → Mild  
  - 2 → Moderate  
  - 3 → Severe  
  - 4 → Proliferative DR  

Split: **70% Train | 15% Validation | 15% Test**  

---

## 🛠️ Tech Stack  
- Python 🐍  
- TensorFlow / Keras  
- OpenCV (for preprocessing)  
- Scikit-Learn (SVM, evaluation metrics)  
- Streamlit / Tkinter (for UI)  

---

## 📊 Model Performance  
| Model            | Accuracy | Highlights                        |
|------------------|----------|-----------------------------------|
| GoogLeNet + SVM  | ~73%     | Lightweight, fast                 |
| ResNet + SVM     | ~74%     | Stronger No-DR detection          |
| Custom CNN       | ~79%     | Best overall classification       |

---

## 🔮 Future Work  
- Integrating **Explainable AI (SHAP)** for deeper interpretability.  
- Using **GANs for Data Augmentation** to handle imbalance.  
- Exploring **Multimodal Learning** (retinal images + patient data).  
- Deploying in **real-world clinical settings**.  

---

## 🏆 Authors  
👤 **Bhagirath Joshi**  
👤 **Meetkumar Sudra**  
(Mentored by *Prof. Dr. Yogesh Naik*)  

---

✨ *“AI won’t replace doctors, but doctors using AI will replace those who don’t.”*  
