import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.models import load_model
from tensorflow.python.tools import module_util as _module_util
import streamlit as st 
import yfinance as yf

st.set_page_config(layout="centered")



st.title("LSTM vs. Random Forest Model Comparison")
st.write("Welcome to our web app for comparing LSTM and Random Forest models in Stock Market Prediction.")

st.header("Introduction")

# Brief Description
st.write("In this web app, we compare the performance of LSTM (Long Short-Term Memory) and Random Forest models on a specific task. We provide tools for loading data, training models, and evaluating their performance. You can explore the differences between these two popular machine learning algorithms and gain insights into which one is better suited for your dataset.")

# Purpose
st.write("The main goal of this app is to help you make informed decisions when choosing between LSTM and Random Forest models for your specific prediction or regression task. You can upload your own dataset or use sample data to see how these models perform.")

# # Instructions
# st.write("To get started, follow these steps:")
# st.write("1. Upload your dataset or use the sample data provided.")
# st.write("2. Select the model (LSTM or Random Forest) you want to compare.")
# st.write("3. Train the selected model and evaluate its performance.")
# st.write("4. Analyze the results and gain insights into model performance.")

# About LSTM and Random Forest
st.write("LSTM is a type of recurrent neural network (RNN) known for its ability to capture sequential patterns, making it suitable for time series data and sequence prediction tasks.")
st.write("Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions, offering versatility and robustness for various machine learning problems.")


import streamlit as st


# st.subheader("LSTM Model")
# st.image("LSTM.png")


# st.subheader("Random Forest Regression")
# st.image("Random Forest.jpg")


col1, col2= st.columns(2)

with col1:
   st.write("**LSTM Model Architecture**")
   st.image("LSTM.png")

with col2:
   st.write("**Random Forest Regression Architecture**")
   st.image("Random Forest.jpg")


# LSTM Section
st.header("LSTM Model")

# Brief Description
st.write("In this section, we'll explore the LSTM (Long Short-Term Memory) model and its characteristics. LSTM is a type of recurrent neural network (RNN) known for its ability to capture sequential patterns and long-range dependencies in data. It's widely used for time series forecasting, natural language processing, and various other sequence-based tasks.")

# LSTM Explanation
st.write("LSTM excels at tasks that involve sequences or time series data because it can retain information over long sequences, unlike traditional feedforward neural networks. It achieves this by using a specialized architecture with memory cells and gates that control the flow of information.")

st.write("*This animation breifly explains the working of the model*")
st.warning("*Source: https://www.youtube.com/watch?v=8HyCNIVRbSU*")

#displaying a local video file

video_file = open('LSTM vid.mp4', 'rb') #enter the filename with filepath

video_bytes = video_file.read() #reading the file

st.video(video_bytes) #displaying the video


# LSTM Section
st.header("Random Forest Regression Model")

# Brief Description
st.write("In this section, we'll explore the Random Forest Regression model, a powerful machine learning technique for regression tasks. Random Forest is an ensemble learning algorithm that combines the predictions of multiple decision trees to provide robust and accurate regression results. It's widely used in various domains, including finance, healthcare, and environmental science.")

# Random Forest Explanation
st.write("Random Forest Regression is particularly well-suited for regression problems where the relationship between input features and the target variable is complex or nonlinear. It leverages the collective wisdom of multiple decision trees to make accurate predictions by reducing overfitting and improving generalization.")

st.write("*This animation breifly explains the working of the model*")
st.warning("*Source: https://www.youtube.com/watch?v=3LQI-w7-FuE&t=1452s*")

#displaying a local video file

video_file = open('RF.mp4', 'rb') #enter the filename with filepath

video_bytes = video_file.read() #reading the file

st.video(video_bytes) #displaying the video



# Project Application
st.header("Prediction Application")

st.write("The following section contains the implementation of both models to give us stock predictions. Graphs will be automatically adjusted for the input stock.")


user_input = st.text_input('Enter Stock Ticker', 'AAPL')
data = yf.download( user_input, start = '2010-01-01', end='2019-12-31')

try:
    data = yf.download(user_input, start='2010-01-01', end='2019-12-31')
except Exception as e:
    st.error(f"An error occurred: {e}")


#describing data
st.subheader('Data from 2010-2019')
st.write(data.describe())

#visulizations 
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(data.Close)
st.pyplot(fig)

#visulizations 
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

#visulizations 
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(data.Close, 'b')
st.pyplot(fig)

# Splitting Data into training and testing

data_training = data['Close'][0:int(len(data) * 0.70)]
data_testing = data['Close'][int(len(data) * 0.70): int(len(data))]

# # Replace NaN values with 0 (or any other appropriate value)
# data_training.fillna(0, inplace=True)
# data_testing.fillna(0, inplace=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))



#load my model 
model = load_model('keras_model.h5')


# testing part
past_100_days = data_training.tail(100)

# # Assuming past_100_days and data_testing are Series objects
# final_df = pd.DataFrame({
#     'past_100_days': past_100_days,
#     'data_testing': data_testing
# })

final_df = past_100_days._append(data_testing, ignore_index=True)

# input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

# print("Input Data: ", input_data)

x_test =[]
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

#making Predictions
y_predicted = model.predict(x_test)


scaler = scaler.scale_

scale_factor = 1/scaler[0]

# print("Scale Factor: ", scale_factor)
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph
st.subheader('LSTM: Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)




# Random Forest Implementation

# Training Model
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)


# Testing Model
x_test =[]
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


# Model Implementation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Reshape the training and testing data for Random Forest
x_train_rf = x_train.reshape(x_train.shape[0], -1)
x_test_rf = x_test.reshape(x_test.shape[0], -1)

y_train_rf = y_train.ravel()  # Convert to 1D array
y_test_rf = y_test.ravel()    # Convert to 1D array

# Create and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(x_train_rf, y_train_rf)

# Make predictions
rf_predictions = rf_model.predict(x_test_rf)

# # Calculate RMSE for Random Forest
# rf_rmse = np.sqrt(mean_squared_error(y_test_rf, rf_predictions))
# print("Random Forest RMSE:", rf_rmse)


# Make predictions with Random Forest
y_predicted_rf = rf_model.predict(x_test_rf)


y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph
st.subheader('Random Forest Regressor: Predictions vs Original')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test_rf, 'b', label = 'Original Price')
plt.plot(y_predicted_rf, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
# plt.show()
st.pyplot(fig3)


