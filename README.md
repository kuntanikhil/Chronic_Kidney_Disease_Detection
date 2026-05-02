# 🩺 Chronic Kidney Disease Detection using Machine Learning

A Machine Learning based web application that predicts whether a patient is suffering from **Chronic Kidney Disease (CKD)** using various medical input parameters such as blood pressure, albumin, sugar level, hemoglobin, serum creatinine, hypertension, diabetes mellitus, and other clinical features.

This project helps in **early diagnosis of CKD**, enabling healthcare professionals and users to take preventive medical action at an early stage.

---

## 📌 Project Overview

Chronic Kidney Disease is a serious long-term medical condition in which the kidneys gradually lose their ability to function properly. Early identification of CKD can significantly reduce health risks and treatment costs.

This project uses **supervised machine learning algorithms** trained on CKD medical datasets to classify whether the patient has:

- **CKD Present**
- **CKD Not Present**

The model analyzes multiple clinical attributes and returns a prediction instantly through a user-friendly web interface.

---

## 🎯 Objectives

- To build an intelligent disease prediction system for CKD.
- To assist in early diagnosis using patient health parameters.
- To reduce manual dependency on lengthy medical analysis.
- To provide fast and accurate prediction results.
- To demonstrate the use of Machine Learning in healthcare.

---

## 🧠 Machine Learning Workflow

The project follows the complete machine learning pipeline:

1. Data Collection
2. Data Cleaning & Preprocessing
3. Handling Missing Values
4. Feature Encoding
5. Model Training
6. Model Evaluation
7. Prediction System Development
8. Web Application Integration

---

## 📂 Dataset Information

The dataset contains various patient medical records used for CKD classification.

### Important Attributes Used:

- Age
- Blood Pressure
- Specific Gravity
- Albumin
- Sugar
- Red Blood Cells
- Pus Cell
- Blood Glucose Random
- Blood Urea
- Serum Creatinine
- Sodium
- Potassium
- Hemoglobin
- Packed Cell Volume
- White Blood Cell Count
- Red Blood Cell Count
- Hypertension
- Diabetes Mellitus
- Appetite
- Pedal Edema
- Anemia

---

## ⚙️ Technologies Used

### Programming Language
- Python

### Libraries Used
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle / Joblib

### Web Framework
- Flask / Django *(update according to your project)*

### Frontend
- HTML
- CSS
- Bootstrap
- JavaScript

---

## 🤖 Machine Learning Algorithms Used

This project involves training and testing of multiple ML models such as:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine
- K-Nearest Neighbors

Among them, the best performing model is selected based on accuracy.

---

## 📈 Model Performance

The trained model achieved high prediction accuracy after preprocessing and feature engineering.

Evaluation metrics used:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🖥️ System Architecture

```text
Patient Input Data
       ↓
Data Preprocessing
       ↓
Trained ML Model
       ↓
Prediction Engine
       ↓
Web Interface Output
```

---

## 📁 Project Structure

```bash
Chronic_Kidney_Disease_Detection/
│
├── dataset/
│   └── kidney_disease.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── static/
│   ├── css/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── model/
│   └── ckd_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run This Project Locally

### Step 1: Clone Repository

```bash
git clone https://github.com/kuntanikhil/Chronic_Kidney_Disease_Detection.git
cd Chronic_Kidney_Disease_Detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Application

```bash
python app.py
```

### Step 5: Open Browser

```bash
http://127.0.0.1:5000/
```

---

## 💡 Features

- Simple and user-friendly interface
- Instant CKD prediction
- Efficient ML based diagnosis
- Medical parameter based intelligent analysis
- Useful healthcare assistance tool

---

## 📸 Output Screenshots

Add your project screenshots here:

- Home Page
- Input Form
- Prediction Result Page

---

## 🔮 Future Enhancements

- Deploying project on cloud
- Adding user authentication
- Integrating real hospital datasets
- Improving prediction accuracy with deep learning
- Doctor recommendation module

---

## 📚 Real World Applications

- Hospitals
- Diagnostic Centers
- Healthcare Startups
- Preventive Medical Screening Systems

---

## 👨‍💻 Author



GitHub: https://github.com/kuntanikhil

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub.
