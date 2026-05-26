import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

petrol_df = pd.read_csv('data/petrol.csv')
diesel_df = pd.read_csv('data/diesel.csv')
print(petrol_df.head())
print(diesel_df.head())
print(petrol_df.info())
print(diesel_df.info())
petrol_df.dropna(inplace=True)
diesel_df.dropna(inplace=True)
petrol_df['date'] = pd.to_datetime(
    petrol_df['date']
)

diesel_df['date'] = pd.to_datetime(
    diesel_df['date']
)
petrol_df = petrol_df.sort_values('date')

diesel_df = diesel_df.sort_values('date')

cities = [
    'Chennai',
    'Delhi',
    'Mumbai',
    'Bangalore'
]

petrol_df['city'] = [
    cities[i % len(cities)]
    for i in range(len(petrol_df))
]

diesel_df['city'] = [
    cities[i % len(cities)]
    for i in range(len(diesel_df))
]

'''petrol_chennai = petrol_df[
    petrol_df['city'] == 'Chennai'
]

diesel_chennai = diesel_df[
    diesel_df['city'] == 'Chennai'
]'''
petrol_df.to_csv(
    "petrol_forecast.csv",
    index=False
)

diesel_df.to_csv(
    "diesel_forecast.csv",
    index=False
)
# SELECT CITY
selected_city = "Chennai"

# FILTER CITY DATA
petrol_chennai = petrol_df[
    petrol_df['city'] == selected_city
]

diesel_chennai = diesel_df[
    diesel_df['city'] == selected_city
]
# petrol visualization
plt.figure(figsize=(14,6))

plt.plot(
    petrol_chennai['date'],
    petrol_chennai['rate']
)

plt.title('Petrol Price Trend')

plt.xlabel('Date')
plt.ylabel('Price')

plt.grid(True)

plt.show()
#diesel visualization
plt.figure(figsize=(14,6))

plt.plot(
    diesel_chennai['date'],
    diesel_chennai['rate']
)

plt.title('Diesel Price Trend')

plt.xlabel('Date')
plt.ylabel('Price')

plt.grid(True)

plt.show()
''' Prophet requires:
->ds → date
->y → target value'''

# petrol forecast preparation
petrol_forecast = petrol_chennai[
    ['date', 'rate']
]
petrol_forecast.columns = ['ds', 'y']

# diesel forecast preparation
diesel_forecast = diesel_chennai[
    ['date', 'rate']
]
diesel_forecast.columns = ['ds', 'y']

petrol_train = petrol_forecast[:-30]
petrol_test = petrol_forecast[-30:]

diesel_train = diesel_forecast[:-30]
diesel_test = diesel_forecast[-30:]

petrol_model = Prophet()
diesel_model = Prophet()

petrol_model.fit(petrol_train)
diesel_model.fit(diesel_train)

#Predict next 30 days.
petrol_future = petrol_model.make_future_dataframe(
    periods=30
)
diesel_future = diesel_model.make_future_dataframe(
    periods=30
)

petrol_prediction = petrol_model.predict(
    petrol_future
)
diesel_prediction = diesel_model.predict(
    diesel_future
)

print(
    petrol_prediction[['ds', 'yhat']].tail()
)
print(
    diesel_prediction[['ds', 'yhat']].tail()
)

#Petrol Forecast
petrol_model.plot(petrol_prediction)
plt.show()

#Diesel Forecast
diesel_model.plot(diesel_prediction)
plt.show()

# Evaluate the model(petrol)
petrol_predicted = petrol_prediction[
    ['ds', 'yhat']
].tail(30)

petrol_compare = petrol_test.copy()

petrol_compare['Predicted'] = (
    petrol_predicted['yhat'].values
)

print(
    mean_absolute_error(
        petrol_compare['y'],
        petrol_compare['Predicted']
    )
)

# Evaluate the model(diesel)
diesel_predicted = diesel_prediction[
    ['ds', 'yhat']
].tail(30)

diesel_compare = diesel_test.copy()

diesel_compare['Predicted'] = (
    diesel_predicted['yhat'].values
)

print(
    mean_absolute_error(
        diesel_compare['y'],
        diesel_compare['Predicted']
    )
)

petrol_prediction[
    ['ds', 'yhat']
].to_csv(
    'petrol_orecast.csv',
    index=False
)

diesel_prediction[
    ['ds', 'yhat']
].to_csv(
    'diesel_orecast.csv',
    index=False
)

