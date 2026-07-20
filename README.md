# AI-Based Medical Imaging Diagnosis (RADAR)

This repository contains an AI-assisted medical imaging diagnosis system developed under the **RADAR – Rajalakshmi Advanced Diagnostics and Research** club. The system focuses on detecting **pneumonia from chest X-ray images** using deep learning and explainable AI techniques.

---

## Problem Statement
Radiologists often face delays and missed early diagnoses due to the rapidly increasing volume of medical imaging data. Manual analysis of chest X-rays is time-consuming and can lead to diagnostic fatigue, especially in high-volume clinical environments.

---

## Proposed Solution
This project proposes an **AI-based decision-support system** that:
- Automatically classifies chest X-ray images as **Normal** or **Pneumonia**
- Provides reminder-level confidence scores
- Uses **Grad-CAM** to visually highlight lung regions influencing the prediction
- Assists radiologists without replacing clinical judgment

---

## Key Features
- Transfer Learning using **ResNet50**
- Binary classification (Normal vs Pneumonia)
- Explainable AI using **Grad-CAM**
- Interactive **Streamlit web interface**
- Downloadable heatmap visualizations
- Clinically inspired confidence thresholds and inconclusive zone

---

## Tech Stack
- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Model Architecture:** ResNet50 (Transfer Learning)  
- **Explainability:** Grad-CAM  
- **Image Processing:** OpenCV  
- **Web Interface:** Streamlit  

---

## Project Structure
Medical_Imaging_Diagnosis/
│
├── app.py # Streamlit application
├── train_model.py # Model training & fine-tuning
├── gradcam.py # Explainable AI (Grad-CAM)
├── requirements.txt # Dependencies
├── radar_logo.png # RADAR club logo
├── README.md
│
├── model/
│ └── pneumonia_model.h5 # Trained model
│
├── dataset/ # (Not uploaded due to size)
└── radar_env/ # (Virtual environment - ignored)

---

## How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Train / Fine-tune the Model
python train_model.py

3️⃣ Run the Web Application
python -m streamlit run app.py


The application will open in your browser and is accessible on both desktop and mobile browsers.

 Dataset

Chest X-ray Pneumonia Dataset (Kaggle)

Dataset is not included in this repository due to size constraints
 Disclaimer

This system is designed strictly as a decision-support tool for academic and research purposes. It is not intended for clinical deployment or direct medical diagnosis.

 Organization

RADAR – Rajalakshmi Advanced Diagnostics and Research
Department of AI & Data Science

## Author 
Abiram R
Department of AI & Data Science 
Rajalakshmi Institute of technology

---

## WHAT TO DO NEXT
1. Save this as **`README.md`**
2. Place it in your **project root**
3. Commit & push to GitHub

If you want, next I can:
- 🔹 Optimize README for **Streamlit Cloud**
- 🔹 Shorten it for **college submission**
- 🔹 Add **project screenshots section**
- 🔹 Write **final report or PPT**

Just tell me
