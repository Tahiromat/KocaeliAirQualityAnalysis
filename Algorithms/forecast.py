import math
import numpy as np
import pandas as pd
import plotly.express as px
from prophet import Prophet
import statsmodels.api as smapi
from plotly import graph_objs as go
from prophet.plot import plot_plotly
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class ForecastingAlgorithmsClass:

    def lstm_forecast(st, data, forecast_parameter):
        data = data.filter([forecast_parameter])
        dataset = data.values
        training_data_len = math.ceil(len(dataset) * 0.8)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:training_data_len, :]

        x_train = []
        y_train = []
        step = 100

        for i in range(step, len(train_data)):
            x_train.append(train_data[i-step:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(x_train, y_train, batch_size=128, epochs=1)

        test_data = scaled_data[training_data_len - step : , :]
        x_test = []
        y_test = dataset[training_data_len : , :]

        for i in range(step, len(test_data)):
            x_test.append(test_data[i-step : i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        rms = np.sqrt(np.mean((predictions - y_test))**2)
        st.write(rms)

        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
      
        fig2 = go.Figure()
        fig2.add_trace(go.Line(x=data.index, y=data[forecast_parameter], name="data"))
        fig2.add_trace(go.Line(x=valid.index, y=valid[forecast_parameter], name="validation"))
        fig2.add_trace(go.Line(x=valid.index, y=valid['Predictions'], name="predictions"))
        fig2.layout.update(xaxis_title="Date", yaxis_title=forecast_parameter+" values",title_text=forecast_parameter, xaxis_rangeslider_visible=False, width=800, height=500)
        st.plotly_chart(fig2)
        

    def arima_forecast(st, data, forecasted_param):
        st.write("Under Development")

        # df = data[[forecasted_param]].copy()
        # n = int(len(df) * 0.8)

        # train = df[forecasted_param][:n]
        # test = df[forecasted_param][n:]

        # model = smapi.tsa.arima.ARIMA(train, order=(6, 1, 3))
        # future = model.fit()

        # step = 180
        # fc  = future.forecast(step, alpha=0.1)
        # fc = pd.Series(fc, index=test[:step].index)

        # fig2 = go.Figure()
        # fig2.layout.update(title_text=forecasted_param, xaxis_rangeslider_visible=False, width=800, height=500)
        # # fig2.add_trace(go.Line(x=train.index, y=train[:step], name="data"))
        # # fig2.add_trace(go.Line(x=test.index, y=test[:step], name="test"))
        # fig2.add_trace(go.Line(x=fc.index, y=fc, name="forecast"))
        # st.plotly_chart(fig2)


    def prophet_forecast(st, data, forecast_column_name):
        df_train = data[['Date', forecast_column_name]]     
        df_train = df_train.rename(columns={"Date": "ds", forecast_column_name: "y"})
        period = 365
        
        m = Prophet(changepoint_range=0.95)
        m.fit(df_train)
        future = m.make_future_dataframe(period)
        forecast = m.predict(future)
        
        fig2 = plot_plotly(m, forecast)
        st.plotly_chart(fig2)