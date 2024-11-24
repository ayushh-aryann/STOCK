import streamlit as st
import requests

# Set the title of the web app
st.title("Stock Price Prediction")

# Input fields for stock symbol and number of days
stock_symbol = st.text_input("Enter Stock Symbol:", "AAPL")
days = st.number_input("Number of Days to Predict:", 1, 365, 30)

# Display the prediction results when the button is pressed
if st.button("Predict"):
    try:
        # Make API request to Flask backend (replace with the URL after deployment)
        response = requests.post("https://your-flask-app-url.onrender.com/predict", json={"stock_symbol": stock_symbol, "days": days})
        
        # Check if the request was successful
        if response.status_code == 200:
            predictions = response.json()
            st.write("Predicted Prices:", predictions)
            st.line_chart(predictions)  # Plot the predictions as a line chart
        else:
            st.error(f"Error: {response.json().get('error')}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
