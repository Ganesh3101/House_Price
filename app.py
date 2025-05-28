import streamlit as st
import pandas as pd
import joblib

# Load your trained model (update filename if different)
model = joblib.load("RandomForest_best_model.joblib")

# Selected features
features = [
    'Gr Liv Area', 'Full Bath', 'Bedroom AbvGr', 'TotRms AbvGrd', 'Year Built'
]

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 Ames House Price Prediction")
st.write("Enter the house features to predict the sale price:")

# Collect input values
input_data = {}
for feat in features:
    default = 2000 if feat == 'Gr Liv Area' else 1 if 'Bath' in feat else 6 if feat == 'TotRms AbvGrd' else 2000
    input_data[feat] = st.number_input(f"{feat}", value=default, step=1)

# Predict on button click
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"🏡 Predicted House Price: **${int(prediction):,}**")