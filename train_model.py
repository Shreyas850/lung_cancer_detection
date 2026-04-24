# Lung Cancer Detection - Machine Learning Model Training
# Author: Shreyas ms

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from pickle import dump

# Load the dataset
data = pd.read_csv("lung_cancer_dataset.csv")

# Encode gender (M=1, F=0)
data["GENDER"] = data["GENDER"].map({"M": 1, "F": 0})

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Separate features and target variable
features = data.drop("LUNG_CANCER", axis=1)
target = data["LUNG_CANCER"]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Evaluate model performance
cm = confusion_matrix(y_test, model.predict(x_test))
print("\nConfusion Matrix:\n", cm)

cr = classification_report(y_test, model.predict(x_test))
print("\nClassification Report:\n", cr)

# Save the trained model
with open("lung_cancer.pkl", "wb") as f:
    dump(model, f)

print("\n✅ Model training complete. Model saved as 'lung_cancer.pkl'")
