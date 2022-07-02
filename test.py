'''import pandas as pd
import datetime
from Preprocessing.preprocessing import PreprocessingClass as PRPC

data = pd.read_excel('/home/tahir/Documents/DataScience/KocaeliAirQuality/TEST/test_data.xlsx')
PRPC.delete_unnecessary_rows(data)
PRPC.change_data_type(data)
# PRPC.change_dataset_index(data)

print(data.head(10))

l2 = []

last_datetime = data['Date'].tail(1).to_string(index=False)
print(last_datetime)

def generate_datetimes(date_from_str=last_datetime, days=30):
   date_from = datetime.datetime.strptime(date_from_str, '%Y-%m-%d %H:%M:%S')
   for hour in range(24*days):
       yield date_from + datetime.timedelta(hours=hour+1)
    
for date in generate_datetimes():
    l2.append(date)
    # print(date.strftime('%Y-%m-%d %H:%M:%S'))

t2 = {"Date":l2}
df2 = pd.DataFrame(t2)




df_new = pd.concat([data, df2], ignore_index=True)

df_new.to_csv('/home/tahir/Documents/DataScience/KocaeliAirQuality/r.csv')

print(df_new)
'''


import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore")

# import plotly.plotly as py
# import plotly.offline as pyoff
import plotly.graph_objs as go


data = pd.read_csv('/home/tahir/Documents/DataScience/HavaKalitesiAnomaliTespiti/Dataset/Adana/adana__çatalan.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date').resample('M').mean().reset_index()
df = data[['Date', 'PM10 ( µg/m3 )']]
df.set_index('Date', inplace=True)

print(df)

train = df

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)

n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)


model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dropout(0.15))
model.add(Dense(1))

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')



history = model.fit_generator(generator,epochs=150,verbose=1)


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plot_data = [
    go.Scatter(
        x=hist['epoch'],
        y=hist['loss'],
        name='loss'
    ),
    
]

plot_layout = go.Layout(
        title='Training loss'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,13) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),index=future_dates[-n_input:].index, columns=['Prediction'])

df_proj = pd.concat([df,df_predict], axis=1)

print(df_proj.head(10))

plot_data = [
    go.Scatter(
        x=df.index,
        y=df_proj['PM10 ( µg/m3 )'],
        name='actual'
    ),
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Prediction'],
        name='prediction'
    )
]

plot_layout = go.Layout(
        title='Shampoo sales prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()