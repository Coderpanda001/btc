import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime

# Load Model 
model = load_model("Bitcoinpredict.keras")

# Function to predict future prices
def predict_future_prices(data, model, scaler, base_days=100, future_days=5):
    m = data
    z = []
    for i in range(base_days, len(m) + future_days):
        m = m.reshape(-1, 1)
        inter = [m[-base_days:, 0]]
        inter = np.array(inter)
        inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
        pred = model.predict(inter)
        m = np.append(m, pred)
        z = np.append(z, pred)
    z = np.array(z)
    z = scaler.inverse_transform(z.reshape(-1, 1))
    return z

# Streamlit app
def main():
    st.header('Bitcoin Price Prediction Model')
    st.subheader('Bitcoin Price Data')
    
    # Date range selector for selecting the data range to display
    start_date = st.date_input("Start Date", datetime(2015, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 11, 30))
    
    # Fetch data using Yahoo Finance API
    data = pd.DataFrame(yf.download('BTC-USD', start=start_date,
end=end_date))
    data = data.reset_index()
    st.write(data)
    
    # Line chart for Bitcoin prices
    st.subheader('Bitcoin Line Chart')
    data.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
    st.line_chart(data)
    
    # Split data into train and test sets
    train_data = data[:-100]
    test_data = data[-200:]
    
    # Scale data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scale = scaler.fit_transform(train_data)
    test_data_scale = scaler.transform(test_data)
    
    # Prepare data for prediction
    x = []
    y = []
    for i in range(100, test_data_scale.shape[0]):
        x.append(test_data_scale[i - 100:i])
        y.append(test_data_scale[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    # Predictions
    pred = model.predict(x)
    pred = scaler.inverse_transform(pred)
    preds = pred.reshape(-1, 1)
    ys = scaler.inverse_transform(y.reshape(-1, 1))
    preds = pd.DataFrame(preds, columns=['Predicted Price'])
    ys = pd.DataFrame(ys, columns=['Original Price'])
    chart_data = pd.concat((preds, ys), axis=1)
    
    st.subheader('Predicted vs Original Prices ')
    st.write(chart_data)
    
    # Line chart for predicted vs original prices
    st.subheader('Predicted vs Original Prices Chart ')
    st.line_chart(chart_data)
    
    # Predict future prices
    future_prices = predict_future_prices(y[-100:], model, scaler)
    
    # Line chart for predicted future prices
    st.subheader('Predicted Future Days Bitcoin Price')
    st.line_chart(future_prices)
    
    # High/Low suggestion
    suggested_action = "High" if future_prices[-1] > future_prices[-2] else "Low"
    
    # Display suggested action in colored box
    st.subheader('Suggested Action')
    if suggested_action == "High":
        st.success(f"Suggested Action: {suggested_action}")
    else:
        st.error(f"Suggested Action: {suggested_action}")
    
    # Display MarketWatch queriesst.markdown("---")
    st.markdown("## Queries")
    st.markdown("Contact / support:")

    # Define the URL
    url = "https://tradelitcare.streamlit.app"

    # Create the button and add the redirection logic
    if st.button('Contact / Support Us'):
        st.markdown(f"[Click here to visit our support site]({url})")
        st.experimental_rerun()
        # Horizontal scrolling disclaimer text
        st.markdown("---")
        st.write(
        """
        <div style="overflow-x: auto; white-space: nowrap;">
            <marquee behavior="scroll" direction="left" scrollamount="5">
                Intraday Data provided by FACTSET and subject to terms of use. 
                Historical and current end-of-day data provided by FACTSET. 
                All quotes are in local exchange time. Real-time last sale data for U.S. 
                stock quotes reflect trades reported through Nasdaq only. 
                Intraday data delayed at least 15 minutes or per exchange requirements.
            </marquee>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
