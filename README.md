# Face Classification and PCA Reconstruction (KNN Classifier)

This project implements a **machine learning pipeline** for **image classification** and **dimensionality reduction** using **K-Nearest Neighbors (KNN)** and **Principal Component Analysis (PCA)**.

---

## 🎯 Objectives
- Perform **classification of face images** from a dataset (`TP1.npy`)
- Apply **data preprocessing and standardization**
- Evaluate model performance using **accuracy and confusion matrix**
- Compare classification **before and after PCA**
- Reconstruct images from **PCA compressed representations**
- Measure **compression error** and **classification speed improvement**

---

## 🧠 Techniques & Tools
- **Language:** Python 3  
- **Libraries:** NumPy, scikit-learn, Matplotlib, Pillow, scikit-image  
- **Algorithms:**  
  - KNN (Euclidean / Manhattan distance)  
  - PCA (Dimensionality reduction)  

---

## 🧩 Key Results
- Best accuracy: **~95%** with KNN (k=3)
- PCA reduced features from 2914 → 100 while maintaining performance
- Classification time reduced significantly using PCA
- Successful reconstruction of original images with low error

---

## 🧪 Features
- **Plot histograms** of class distribution  
- **Visualize sample images and PCA eigenfaces**  
- **Predict new images** (e.g. `Bush.jpg`)  
- **Compare accuracy across different k values**

---

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python TP2_ETU.py
