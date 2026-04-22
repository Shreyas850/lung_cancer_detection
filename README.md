# 🫁 Lung Cancer Detection using Machine Learning

> 🗓️ This repository contains Week 2 submission for AICTE Internship — focusing on Machine Learning model training.


This project aims to detect the likelihood of **lung cancer** in individuals based on various health and lifestyle parameters such as smoking habits, anxiety, fatigue, and chest pain.

---

## 📊 Dataset
**Source:** [Kaggle - Lung Cancer Dataset](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer)

File used: `lung_cancer_dataset.csv`

---

## 🧠 Machine Learning Model
- **Algorithm:** Logistic Regression  
- **Framework:** Scikit-learn  
- **Language:** Python  
- **Output:** Trained model saved as `lung_cancer.pkl`

---

## ⚙️ Steps in the Code
1. Load and preprocess dataset  
2. Encode categorical columns (e.g., Gender)  
3. Split dataset into train and test sets  
4. Train Logistic Regression model  
5. Evaluate using confusion matrix and classification report  
6. Save the trained model as `lung_cancer.pkl`

---

## 🧪 Run Locally
```bash
# Clone the repository
git clone https://github.com/<your-username>/lung-cancer-detection-ml.git
cd lung-cancer-detection-ml

# Install dependencies
pip install pandas scikit-learn

# Run the training script
python train_model.py


👩‍💻 Author

Shreyas.ms
Second-Year Computer Engineering Student
JSS Pre-University college in Banglore