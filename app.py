import streamlit as st
import pandas as pd
import joblib
import time as t

model = joblib.load("LinearRegression_best_model.joblib")
label_encoder_city = joblib.load("label_encoder_city.joblib")
scaler = joblib.load("scaled.joblib")

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction")
st.image("static/house_price.jpg")
st.write("Enter the house features to predict the sale price:")

df = pd.read_csv("static/data.csv", header=0)
df.dropna(inplace=True, axis=1)
df.drop(columns = ['date','yr_renovated','street','country','statezip','condition','sqft_basement','waterfront','view'],inplace=True)

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above']
input_data = {}
for feat in features:
    default = 1 if feat in ['bedrooms', 'bathrooms'] else 1000 if feat == 'sqft_living' else 1000 if feat == 'sqft_above' else 4000 if feat == 'sqft_lot' else 0
    input_data[feat] = st.number_input(f"{feat}", value=default, step=1)

st.title("Select Floor")
floor_list = df['floors'].unique().tolist()
selected_floor = st.selectbox("Select a floor", floor_list)

st.title("Select City")
city_list = df['city'].unique().tolist()
selected_city = st.selectbox("Select a city", city_list)
encoded_city = label_encoder_city.transform([selected_city])[0]

st.title('Choose Year')
selected_year = st.slider("Select a year", min_value=1900, max_value=2014, step=1)

model_input = {
    'bedrooms': input_data['bedrooms'],
    'bathrooms': input_data['bathrooms'],
    'sqft_living': input_data['sqft_living'],
    'sqft_lot': input_data['sqft_lot'],
    'floors': selected_floor,
    'sqft_above': input_data['sqft_above'],
    'yr_built': selected_year,
    'city': encoded_city
}
if st.button("Predict House Price"):
    input_df = pd.DataFrame([model_input])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    with st.spinner("predicting results..."):
        t.sleep(3)

    st.balloons()
    st.success(f"🏡 Predicted House Price: **${int(prediction[0]*1000):,}**")