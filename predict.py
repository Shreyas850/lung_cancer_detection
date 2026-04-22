import pickle
import pandas as pd

# 1. Load the trained model
model_path = "lung_cancer.pkl"
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

print(f"System: Model '{model_path}' loaded successfully.\n")

# 2. Define the exact feature names seen in your terminal output
feature_names = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 
    'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
    'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]

# 3. Create a DataFrame instead of a NumPy array
new_patient_data = pd.DataFrame(
    [[1, 65, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2]], 
    columns=feature_names
) 

# 4. Execute prediction
prediction = loaded_model.predict(new_patient_data)

# 5. Output logic
print("--- Diagnostic Prediction ---")
if prediction[0] == "YES": 
    print("Result: High likelihood of Lung Cancer detected.")
else:
    print("Result: Low likelihood of Lung Cancer detected.")
print("---------------------------")