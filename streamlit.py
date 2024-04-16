import streamlit as st 
from datetime import date
import plotly.express as px
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os




import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go 
START = "2015-01-01"
TODAY= date.today().strftime("%Y-%m-%d")
st.title("Jake's Stock Prediction Dashboard📈")

stocks=("TSLA","GOOG")
selected_stocks=st.selectbox("Select dataset for prediction", stocks)
n_years=st.slider("Years of prediction:", 1, 4)
period=n_years*365

def load_data(ticker): 
    data= yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data 

data_load_state=st.text("Load Data...")
data= load_data(selected_stocks)
data_load_state.text("Loading Data...done!")


st.subheader('Raw data')
st.write(data.tail())
st.write()

def plot_raw_data(): 
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='stock_open'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Figure Visualizations 
st.write('**Forecast Data**')
fig1=plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('**Forecast Components**')
fig2=m.plot_components(forecast)
st.write(fig2)

# LSTM Prediction modeluploaded_file = st.file_uploader("Choose a CSV file", type="csv")
st.title('LSTM Prediction')
file_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'tsla_prediction_lstm.csv')
if os.path.exists(file_path):
    # Load the CSV data into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Display the head of the DataFrame
    st.write("Preview of the CSV Data:")
    st.dataframe(data.head())  # By default, 'head()' displays the first 5 rows
else:
    st.error("File not found. Please check the file path.")




# visualization of lstm 
    

    # Convert the Date column to datetime type if it's not already
data['Date'] = pd.to_datetime(data['Date'])

    # Set the date as the index of the dataframe
data.set_index('Date', inplace=True)

    # Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Close'], label='Actual Close Price', marker='o', linestyle='-')
plt.plot(data.index, data['Predictions'], label='Predicted Price', marker='o', linestyle='--')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

    # Display the plot
st.pyplot(plt)
