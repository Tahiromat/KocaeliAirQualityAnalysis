import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from plotly import graph_objs as go
import plotly.express as px

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

import matplotlib.pyplot as plt



class AnomalyDetectionAlgorithmsClass:

    def autoencoder_anomaly(st, data, selected_param):

        st.write("Under Development")
    #     df = data[['Date', selected_param]]
    #     df['Date'] = pd.to_datetime(df['Date'])

    #     train = data[(data['Date'] >= '2014') & (data['Date'] <= '2022-02-01')]
    #     test = data[(data['Date'] > '2022-02-01')]

    #     scaler = StandardScaler()
    #     scaler = scaler.fit(np.array(train[selected_param]).reshape(-1,1))

    #     train[selected_param] = scaler.transform(np.array(train[selected_param]).reshape(-1,1))
    #     test[selected_param] = scaler.transform(np.array(test[selected_param]).reshape(-1,1))

    #     TIME_STEPS=30
      
    #     def create_sequences(X, y, time_steps=TIME_STEPS):
    #         X_out, y_out = [], []
    #         for i in range(len(X)-time_steps):
    #             X_out.append(X.iloc[i:(i+time_steps)].values)
    #             y_out.append(y.iloc[i+time_steps])
            
    #         return np.array(X_out), np.array(y_out)

    #     X_train, y_train = create_sequences(train[[selected_param]], train[selected_param])
    #     X_test, y_test = create_sequences(test[[selected_param]], test[selected_param])

    #     np.random.seed(21)
    #     tf.random.set_seed(21) 

    #     model = Sequential()
    #     model.add(LSTM(128, activation = 'tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
    #     model.add(Dropout(rate=0.2))
    #     model.add(RepeatVector(X_train.shape[1]))
    #     model.add(LSTM(128, activation = 'tanh', return_sequences=True))
    #     model.add(Dropout(rate=0.2))
    #     model.add(TimeDistributed(Dense(X_train.shape[2])))
    #     model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    #     model.summary()

    #     history = model.fit(
    #         X_train,
    #         y_train,
    #         epochs=2,
    #         batch_size=32,
    #         validation_split=0.1,
    #         callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
    #         shuffle=False)


    def isolationforest_anomaly(st, data, selected_param):
        df = data[['Date', selected_param]]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').resample('D').mean().reset_index()
       
        model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.004),max_features=1.0)
        model.fit(df[[selected_param]])
        df['outliers'] = pd.Series(model.predict(df[[selected_param]])).apply(lambda x: 'yes' if (x == -1) else 'no')
        
        fig2 = px.scatter(df.reset_index(), x='Date', y=selected_param, color='outliers')
        fig2.layout.update(title_text=selected_param, xaxis_rangeslider_visible=False, width=800, height=500)
        st.plotly_chart(fig2)



    def prophet_anomaly(st, data, selected_param):
        df = data[['Date', selected_param]]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').resample('D').mean().reset_index()
        
        data = df.reset_index()[['Date', selected_param]].rename({'Date':'ds', selected_param:'y'}, axis='columns')
        train = data[(data['ds'] >= '2014') & (data['ds'] <= '2022-02-01')]
        test = data[(data['ds'] > '2022-02-01')]

        m = Prophet(changepoint_range=0.95)
        m.fit(train)
        future = m.make_future_dataframe(periods=365, freq='D')
        forecast = m.predict(future)
        result = pd.concat([data.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]], axis=1)
        # fig2 = m.plot(forecast)
        result['error'] = result['y'] - result['yhat']
        result['uncertainty'] = result['yhat_upper'] - result['yhat_lower']
        result['anomaly'] = result.apply(lambda x: 'Yes' if(np.abs(x['error']) > 1.5*x['uncertainty']) else 'No', axis = 1)
       
        fig2 = px.line(result.reset_index(), x='ds', y='y', color='anomaly')
        fig2.layout.update(title_text=selected_param, xaxis_rangeslider_visible=False, width=800, height=500)
        st.plotly_chart(fig2)


        