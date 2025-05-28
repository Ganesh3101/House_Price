import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("RandomForest_best_model.joblib")  # Replace with your best model file

# Integer features (from your AmesHousing.csv)
features = [
    'MS SubClass', 'Lot Area', 'Overall Qual', 'Overall Cond',
    'Year Built', 'Year Remod/Add', '1st Flr SF', '2nd Flr SF',
    'Low Qual Fin SF', 'Gr Liv Area', 'Full Bath', 'Half Bath',
    'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces',
    'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch',
    'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold'
]

# Streamlit app
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 Ames House Price Prediction App")
st.write("Enter the property details below to estimate the house sale price.")

# Input form
input_data = {}
for feat in features:
    input_data[feat] = st.number_input(f"{feat}", value=0)

# Predict button
if st.button("Predict Price"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.success(f"🏡 Estimated House Price: **${int(prediction):,}**")