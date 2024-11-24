# stockpriceprediction.py
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math
from sklearn.metrics import mean_squared_error

def fetch_stock_data(symbol, start_date, end_date, api_key):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

def preprocess_data(df):
    df1 = df['close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    return df1, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def predict_stock_prices(stock_symbol, days, api_key="24bf0daaf569a4eb0647309690fe5563001f8fd9"):
    start_date, end_date = "2015-05-27", "2020-05-22"
    df = fetch_stock_data(stock_symbol, start_date, end_date, api_key)
    df.to_csv(f'{stock_symbol}.csv', index=False)
    
    df1, scaler = preprocess_data(df)
    training_size = int(len(df1) * 0.65)
    train_data = df1[0:training_size, :]
    test_data = df1[training_size:, :1]

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict[:len(y_train)]))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
    print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")

    x_input = test_data[-time_step:].reshape(1, -1)
    temp_input = list(x_input[0])
    predictions = []

    for i in range(days):
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1).reshape((1, time_step, 1))
        else:
            x_input = x_input.reshape((1, time_step, 1))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0][0])
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).tolist()
