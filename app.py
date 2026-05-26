import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Fuel Price Forecasting App')

st.write(
    'Forecast Petrol and Diesel Prices'
)

petrol = pd.read_csv(
    'petrol_forecast.csv'
)

diesel = pd.read_csv(
    'diesel_forecast.csv'
)

city = st.selectbox(
    "Select City",    
    petrol['city'].unique()
)

petrol = petrol[petrol['city'] == city]
diesel = diesel[diesel['city'] == city]

st.header(f"Petrol Prices - {city}")

st.write(
    petrol.head()
)

st.header(f"Diesel Prices - {city}")

st.write(
    diesel.head()
)

petrol1 = pd.read_csv(
    'petrol_orecast.csv'
)

diesel1 = pd.read_csv(
    'diesel_orecast.csv'
)

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    petrol1['ds'],
    petrol1['yhat'],
    label='Petrol'
)

ax.plot(
    diesel1['ds'],
    diesel1['yhat'],
    label='Diesel'
)

plt.xticks(rotation=45)

plt.legend()

st.pyplot(fig)