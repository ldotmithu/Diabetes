import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model and preprocessor
model = joblib.load(Path('artifacts/model_train/model.joblib'))
churn_preprocess = joblib.load(Path('artifacts/model_train/preprocess.joblib'))

# Create the Streamlit UI with tabs
st.title("Diabetes Prediction Using Machine Learning")

# Create tabs for "Prediction" and "Feature Explanation"
tabs = st.tabs(["Prediction", "Feature Explanation"])

with tabs[0]:
    st.header("Predict Diabetes Risk")

    # Input fields for the features
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=150, value=30)
    Glucose = st.slider("Glucose", min_value=40, max_value=220, value=120)
    BloodPressure = st.number_input("BloodPressure", min_value=40, max_value=211, value=80)
    SkinThickness = st.slider("SkinThickness", min_value=2, max_value=20, value=2)
    Insulin = st.number_input("Insulin", value=0.0)
    BMI = st.number_input("BMI", value=0.0)
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", value=0.0)
    Age = st.slider("Age", min_value=10, max_value=90, value=20)

    # Create a DataFrame from input data
    input_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age],
    })

    # Preprocess the input data
    try:
        input_data_preprocessed = churn_preprocess.transform(input_data)
    except ValueError as e:
        st.error(f"Error in preprocessing: {e}")
        input_data_preprocessed = None

    # Predict diabetes risk
    if st.button("Predict") and input_data_preprocessed is not None:
        with st.spinner("Processing..."):
            try:
                prediction = model.predict(input_data_preprocessed)[0]
                
                # Display appropriate message based on prediction
                if prediction == 0:
                    st.success("Prediction: **No Diabetes**")
                else:
                    st.success("Prediction: **Diabetes**")
            except Exception as e:
                st.error(f"Error in prediction: {e}")

with tabs[1]:
    st.header("Feature Explanations")

    st.write("""
    ### Feature Explanations:
    - **Pregnancies**: Number of pregnancies.
    - **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test.
    - **BloodPressure**: Diastolic blood pressure (mm Hg).
    - **SkinThickness**: Triceps skinfold thickness (mm).
    - **Insulin**: 2-Hour serum insulin (mu U/ml).
    - **BMI**: Body Mass Index (weight in kg/(height in m)^2).
    - **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
    - **Age**: Age of the individual in years.
    """)
