import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the pre-trained models and preprocessors
model_behavior = joblib.load('model_behavior.pkl')
model_performance = joblib.load('model_performance.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Define function to preprocess input data
def preprocess_input(data):
    # Define the order of categorical and numerical features based on the model fitting
    categorical_features = ['Gender', 'Disability Type', 'Comorbidity', 'Academic Support Received', 'Psychological Therapy']
    numerical_features = ['Adaptation Score', 'Class Attention Score', 'Communication Score',
                           'Health Behaviors', 'Hyperactivity', 'Impulsivity', 'Inattention',
                           'Life Satisfaction', 'Math Score', 'Resilience Score', 'Science Score']
    
    # Create DataFrame from the input data
    df = pd.DataFrame(data)
    
    # Ensure the features are in the correct order
    X_cat = df[categorical_features]
    X_num = df[numerical_features]
    
    # One-hot encode categorical features
    X_cat_encoded = encoder.transform(X_cat)
    
    # Scale numerical features
    X_num_scaled = scaler.transform(X_num)
    
    # Combine preprocessed numerical and categorical features
    X_preprocessed = np.hstack([X_num_scaled, X_cat_encoded])
    
    return X_preprocessed

# Streamlit UI
st.title('Student Performance and Behaviour Prediction')

# User inputs
st.header('Enter Student Details')
gender = st.selectbox('Gender', ['Male', 'Female'])
disability_type = st.selectbox('Disability Type', ['Learning Disability', 'ADHD', 'Autism', 'Physical Disability'])
comorbidity = st.selectbox('Comorbidity', ['Yes', 'No'])
academic_support = st.selectbox('Academic Support Received', ['Yes', 'No'])
psychological_therapy = st.selectbox('Psychological Therapy', ['Yes', 'No'])

adaptation_score = st.slider('Adaptation Score', 1, 10)
health_behaviors = st.slider('Health Behaviors', 1, 5)
resilience_score = st.slider('Resilience Score', 1, 10)
life_satisfaction = st.slider('Life Satisfaction', 1, 10)
class_attention_score = st.slider('Class Attention Score', 1, 10)
math_score = st.slider('Math Score', 50, 100)
science_score = st.slider('Science Score', 50, 100)
communication_score = st.slider('Communication Score', 50, 100)

# ADHD-specific inputs
hyperactivity = st.slider('Hyperactivity', 1, 10)
impulsivity = st.slider('Impulsivity', 1, 10)
inattention = st.slider('Inattention', 1, 10)

# Collect input data
input_data = {
    'Gender': [gender],
    'Disability Type': [disability_type],
    'Comorbidity': [comorbidity],
    'Academic Support Received': [academic_support],
    'Psychological Therapy': [psychological_therapy],
    'Adaptation Score': [adaptation_score],
    'Health Behaviors': [health_behaviors],
    'Resilience Score': [resilience_score],
    'Life Satisfaction': [life_satisfaction],
    'Class Attention Score': [class_attention_score],
    'Inattention': [inattention],
    'Hyperactivity': [hyperactivity],
    'Impulsivity': [impulsivity],
    'Math Score': [math_score],
    'Science Score': [science_score],
    'Communication Score': [communication_score]
}

# Preprocess the input data
X_preprocessed = preprocess_input(pd.DataFrame(input_data))

# Predict with the models
behavior_pred = model_behavior.predict(X_preprocessed)
performance_pred = model_performance.predict(X_preprocessed)

# Display results
st.header('Predictions')
st.write(f"Predicted Behavior Score: {behavior_pred[0]:.2f}")
st.write(f"Predicted Student Performance: {performance_pred[0]:.2f}")



