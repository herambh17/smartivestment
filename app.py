import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
from datetime import date
import math
import yfinance as yf
import streamlit as st
import pickle

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal

from sklearn.metrics import r2_score




start = '2010-01-01'
end =date.today()

st.title('SMART INVEST')
user_input=st.text_input("Enter Stock Ticker",'^NSEBANK')

ticker_symbol = user_input
data = yf.download(ticker_symbol, start, end)
st.subheader('Data from 2010 to Till Date')

st.write(data.head())



st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(20,14))
plt.plot(data.Close)
st.pyplot(fig)

data.rename(columns={'Adj Close':'Price'},inplace=True)
del data['Open']
del data['High']
del data['Low']
del data['Volume']
del data['Close']




def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Price'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['Price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Price'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema']-dataset['26ema']

    # Create Bollinger Bands
    dataset['20sd'] = dataset['Price'].rolling(window = 21).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['Price'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset['Price']-1
    dataset['log_momentum'] = np.log(dataset['momentum'])
    return dataset

df = get_technical_indicators(data)
df = df.dropna()

def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(20, 14), dpi=150)
    shape_0 = dataset.shape[0]
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = list(dataset.index)

    # Create a subplot figure
    fig, ax = plt.subplots(2, 1, figsize=(50, 40))

    # Plot first subplot
    ax[0].plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
    ax[0].plot(dataset['Price'], label='Closing Price', color='b')
    ax[0].plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    ax[0].plot(dataset['upper_band'], label='Upper Band', color='c')
    ax[0].plot(dataset['lower_band'], label='Lower Band', color='c')
    ax[0].fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    ax[0].set_title('Technical indicators for last {} days.'.format(last_days))
    ax[0].set_ylabel('USD')
    ax[0].legend()

    # Plot second subplot
    ax[1].set_title('MACD')
    ax[1].plot(dataset['MACD'], label='MACD', linestyle='-.')
    # ax[1].hlines(15, 0, len(dataset), colors='g', linestyles='--')  # Uncomment if needed
    # ax[1].hlines(-15, 0, len(dataset), colors='g', linestyles='--')  # Uncomment if needed
    ax[1].plot(dataset['log_momentum'], label='Momentum', color='b', linestyle='-')
    ax[1].legend()

    st.subheader('Technical Indicators')# Display the plot in Streamlit
    st.pyplot(fig)

plot_technical_indicators(df, 1000)

split_index = int(len(df) * 0.7)
data_training = df.iloc[:split_index].copy()
data_testing = df.iloc[split_index:].copy()



scalar = MinMaxScaler()
data_training_scaled = scalar.fit_transform(data_training)


custom_objects = {'Orthogonal': Orthogonal}
model = load_model('my_model.h5', custom_objects=custom_objects)


past_60 = data_training.tail(60)
dt = pd.concat([past_60, data_testing], ignore_index=True)
inputs = scalar.fit_transform(dt)
X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
    
X_test, y_test = np.array(X_test), np.array(y_test)


scale = 1/scalar.scale_[0]
y_pred = model.predict(X_test)

y_pred = y_pred*scale
y_test = y_test*scale


r_squared = r2_score(y_test, y_pred)
r_squared_percentage = r_squared * 100


st.subheader(f"Prediction accuracy: {r_squared_percentage:.2f}%")

st.subheader('Actual vs Predicted Batch Size 32')
fig2=plt.figure(figsize=(36,18),dpi=150)
plt.plot(y_test, color = 'red', label = 'Real Bank Nifty Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Bank Nifty Price')
plt.title('Price Prediction-After 10 epochs and Batch Size=32')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


st.subheader('Actual vs Predicted Batch size 64')
fig3=plt.figure(figsize=(36,18),dpi=150)
plt.plot(y_test, color = 'red', label = 'Real Bank Nifty Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Bank Nifty Price')
plt.title('Price Prediction-After 25 epochs and Batch Size=64')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
