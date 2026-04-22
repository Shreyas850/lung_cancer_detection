import streamlit as st
import pickle
import pandas as pd

# Load the model
@st.cache_resource
def load_model():
    with open('lung_cancer.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Define feature names
feature_names = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 
    'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
    'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]

# UI Setup
st.title("Lung Cancer Prediction System")
st.write("Enter patient details below to assess the likelihood of lung cancer based on the trained Logistic Regression model.")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety", ["Yes", "No"])
    peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
    chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])

with col2:
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    allergy = st.selectbox("Allergy", ["Yes", "No"])
    wheezing = st.selectbox("Wheezing", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consuming", ["Yes", "No"])
    coughing = st.selectbox("Coughing", ["Yes", "No"])
    shortness_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
    swallowing_diff = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

# Conversion dictionary: Model expects 2 for Yes/Male, 1 for No/Female
def convert_input(val):
    if val in ["Yes", "Male"]:
        return 2
    return 1

# Prediction Button
if st.button("Predict"):
    # Format the data exactly as the model expects
    input_data = [
        convert_input(gender), age, convert_input(smoking), convert_input(yellow_fingers), 
        convert_input(anxiety), convert_input(peer_pressure), convert_input(chronic_disease), 
        convert_input(fatigue), convert_input(allergy), convert_input(wheezing), 
        convert_input(alcohol), convert_input(coughing), convert_input(shortness_breath), 
        convert_input(swallowing_diff), convert_input(chest_pain)
    ]
    
    df = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(df)
    
    st.markdown("---")
    if prediction[0] == "YES":
        st.error("### Result: High likelihood of Lung Cancer detected.")
    else:
        st.success("### Result: Low likelihood of Lung Cancer detected.")