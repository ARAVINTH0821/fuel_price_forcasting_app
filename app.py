import streamlit as st
import pandas as pd
import numpy as np
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
    petrol.tail(5)
)

st.header(f"Diesel Prices - {city}")

st.write(
    diesel.tail(5)
)

# Convert date column
petrol["ds"] = pd.to_datetime(petrol["ds"], errors="coerce")
petrol = petrol.dropna(subset=["ds"])
petrol["year"] = petrol["ds"].dt.year

petrol_yearly = (
    petrol.groupby("year")["yhat"]
    .mean()
    .reset_index()
)

# Diesel data
diesel["ds"] = pd.to_datetime(diesel["ds"], errors="coerce")
diesel = diesel.dropna(subset=["ds"])
diesel["year"] = diesel["ds"].dt.year

diesel_yearly = (
    diesel.groupby("year")["yhat"]
    .mean()
    .reset_index()
)

# Merge data
combined = pd.merge(
    petrol_yearly,
    diesel_yearly,
    on="year",
    suffixes=("_petrol", "_diesel")
)

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(combined["year"]))
width = 0.4

ax.bar(
    x - width/2,
    combined["yhat_petrol"],
    width,
    label="Petrol",
    color="blue"
)

ax.bar(
    x + width/2,
    combined["yhat_diesel"],
    width,
    label="Diesel",
    color="orange"
)

ax.set_title("Average Fuel Rate by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Average Fuel Rate")
ax.set_xticks(x)
ax.set_xticklabels(combined["year"])
ax.legend()

plt.tight_layout()
#plt.show()

st.pyplot(fig)