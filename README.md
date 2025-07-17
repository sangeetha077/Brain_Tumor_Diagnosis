# 🧠 Brain Disease Diagnosis using BAT Algorithm

## 📌 Project Overview
This project presents a robust system for detecting and classifying brain tumors  from MRI images. The diagnosis system combines **GLCM-based feature extraction**, **BAT algorithm for feature selection**, and **SVM classification** to improve diagnostic accuracy.

---

## 🧠 Abstract
Brain tumor detection in MRIs is difficult due to heterogeneous signals and limited training data. This project addresses those challenges by:
- Segmenting MR images into **superpixels**
- Extracting features using **GLCM**
- Selecting optimal features using the **Bat Algorithm**
- Classifying results using **Support Vector Machine (SVM)**

---

## 🛠️ Technologies and Techniques
- **Language:** Python
- **Libraries:** OpenCV, NumPy, scikit-learn, matplotlib
- **Image Processing:** Median Filtering, Grayscale Conversion
- **Feature Extraction:** GLCM (Contrast, Energy, Homogeneity, Correlation)
- **Feature Selection:** BAT (Bat-Inspired Optimization Algorithm)
- **Classification:** SVM (with RBF Kernel)

---

## ⚙️ Modules Breakdown

### 1. 📷 Image Preprocessing
- Normalize size, contrast, and brightness
- Remove noise and artifacts using median filters

### 2. 📐 Feature Extraction
- Use GLCM for texture-based features
- Convert complex images into useful feature vectors

### 3. 🦇 Feature Selection using BAT Algorithm
- Optimize the feature set by simulating bat echolocation behavior
- Improve model speed and accuracy

### 4. 🧪 Classification with SVM
- Use selected features to classify MRI images into:
  - 🟢 Normal (No tumor)
  - 🔴 Abnormal (Tumor detected)

---

## 🧪 Experimental Results

| Technique Used     | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| GLCM Only          | 88.3%    | -         | -      | 87.2%    |
| GLCM + BAT         | 92.7%    | -         | -      | 91.9%    |
| GLCM + BAT + SVM   | **94.8%**| **94.0%**  | 94.5%  | **94.1%**|

---

## 📈 System Architecture

```mermaid
graph LR
A[Input MRI Images] --> B[Preprocessing]
B --> C[GLCM Feature Extraction]
C --> D[BAT Algorithm - Feature Selection]
D --> E[SVM Classification]
E --> F[Diagnosis Output: Tumor / No Tumor]
