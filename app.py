from flask import Flask, request, jsonify
from flask_cors import CORS
from stockpriceprediction import predict_stock_prices

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/')
def home():
    return "Stock Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    stock_symbol = data.get('stock_symbol', 'AAPL')  # Default to AAPL
    days = data.get('days', 30)  # Default to 30 days
    
    try:
        predictions = predict_stock_prices(stock_symbol, days)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
