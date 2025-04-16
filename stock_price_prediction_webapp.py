import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "AAPL")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

apple_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(apple_data)

# Split the data into training and testing set but I have already imported the trained model "Latest_stock_price_model.keras"
# So only test data set
splitting_len = int(len(apple_data)*0.7)
x_test = pd.DataFrame(apple_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
apple_data['MA_for_250_days'] = apple_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), apple_data['MA_for_250_days'],apple_data,0))

st.subheader('Original Close Price and MA for 200 days')
apple_data['MA_for_200_days'] = apple_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), apple_data['MA_for_200_days'],apple_data,0))

st.subheader('Original Close Price and MA for 100 days')
apple_data['MA_for_100_days'] = apple_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), apple_data['MA_for_100_days'],apple_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), apple_data['MA_for_100_days'],apple_data,1,apple_data['MA_for_250_days']))

# Pre-process the test data first
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# To calculate the accuracy
x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = apple_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([apple_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)


st.subheader("Future Price values")
# st.write(ploting_data)

last_100 = apple_data[['Adj Close']].tail(100)
last_100 = scaler.fit_transform(last_100['Adj Close'].values.reshape(-1,1)).reshape(1,-1,1)
prev_100 = np.copy(last_100).tolist()

def predict_future(no_of_days, prev_100):
    future_predictions = []    
    prev_100 = np.array(prev_100)
    if prev_100.ndim == 2:
        prev_100 = prev_100.reshape(1, prev_100.shape[0], prev_100.shape[1])
    elif prev_100.ndim == 1:
        prev_100 = prev_100.reshape(1, prev_100.shape[0], 1)

    for i in range(no_of_days):
        next_day = model.predict(prev_100)        
        if isinstance(next_day, np.ndarray):
            next_day = next_day.tolist()
        future_predictions.append(scaler.inverse_transform([next_day[0]]))
        
        new_data_point = np.array(next_day[-1]).reshape(1, 1, -1)
        prev_100 = np.concatenate((prev_100[:, 1:, :], new_data_point), axis=1)
        
    return future_predictions

no_of_days = int(st.text_input("Enter the No of days to be predicted from current date : ","10"))
future_results = predict_future(no_of_days,prev_100)
future_results = np.array(future_results).reshape(-1,1)
print(future_results)
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_results), marker = 'o')
for i in range(len(future_results)):
    plt.text(i, future_results[i], f'{future_results[i][0]:,.2f}')
plt.xlabel('days')
plt.ylabel('Close Price')
plt.xticks(range(no_of_days))
y_min = np.min(future_results)
y_max = np.max(future_results)
y_step = (y_max - y_min) / 10 if (y_max - y_min) > 10 else 1
plt.yticks(np.arange(int(y_min), int(y_max) + y_step, y_step))
plt.title('Closing Price of Apple')
st.pyplot(fig)