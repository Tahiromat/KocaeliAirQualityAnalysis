from cProfile import label
import numpy as np
import pandas as pd
from tensorflow import keras
import plotly.graph_objs as go
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Bidirectional
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import warnings
warnings.filterwarnings("ignore")

'''data = pd.read_csv('/home/tahir/Documents/DataScience/KocaeliAirQuality/TEST/İstanbulAverageDF.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date').resample('M').mean().reset_index()
df = data[['Date', 'PM10 ( µg/m3 )']]
df.set_index('Date', inplace=True)
train = df

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)

n_input = 24
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=128)

model = Sequential()
model.add(LSTM(units =200, activation='relu', return_sequences=True, input_shape = (n_input,1)))
model.add(Dropout(0.15))
model.add(LSTM(units =50, activation='relu', return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(units =50, activation='relu', return_sequences=False))
model.add(Dropout(0.15))
model.add(Dense(units=1)) # Prediction of the next value

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mean_absolute_error','acc'])
model.summary()
history = model.fit_generator(generator,epochs=100,verbose=1)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mean_absolute_error'], name='mean_absolute_error'))
fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['acc'], name='accuracy'))
fig.show()


pred_list = []
batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,25) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),index=future_dates[-n_input:].index, columns=['Prediction'])
df_proj = pd.concat([df,df_predict], axis=1)

print(df_proj.tail(25))


fig = go.Figure()
fig.add_trace(go.Line(x=df.index, y=df_proj['PM10 ( µg/m3 )'], name='actual'))
fig.add_trace(go.Line(x=df_proj.index, y=df_proj['Prediction'], name='prediction'))

fig.show()



'''


data = pd.read_csv('/home/tahir/Documents/DataScience/KocaeliAirQuality/TEST/İstanbulAverageDF.csv',   parse_dates=['Date'])
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date').resample('M').mean().reset_index()
df = data[['Date', 'PM10 ( µg/m3 )']]
df.set_index('Date', inplace=True)



# df['hour'] = df.index.hour
# df['day_of_month'] = df.index.day
# df['day_of_week'] = df.index.dayofweek
# df['month'] = df.index.month

# print(df)
# grouped_df = df.groupby('month').mean()
# print(grouped_df)


# fig = go.Figure()
# fig.add_trace(go.Line(x=grouped_df.index, y=grouped_df['PM10 ( µg/m3 )']))
# fig.show()


train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

# f_columns = ['PM10 ( µg/m3 )']

# f_transformer = RobustScaler()
# f_transformer = f_transformer.fit(train[f_columns].to_numpy())
# train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
# test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())


cnt_transformer = MinMaxScaler()
cnt_transformer = cnt_transformer.fit(train[['PM10 ( µg/m3 )']])
train['PM10 ( µg/m3 )'] = cnt_transformer.transform(train[['PM10 ( µg/m3 )']])
test['PM10 ( µg/m3 )'] = cnt_transformer.transform(test[['PM10 ( µg/m3 )']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 12
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train, train['PM10 ( µg/m3 )'], time_steps)
X_test, y_test = create_dataset(test, train['PM10 ( µg/m3 )'], time_steps)
print(X_train.shape, y_train.shape)

model = Sequential()
model.add(Bidirectional(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(rate=0.2))
model.add(LSTM(units =50, activation='relu', return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(units =50, activation='relu', return_sequences=False))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mean_absolute_error'])


# model = Sequential()
# model.add(
#     Bidirectional(
#         LSTM(
#         units=128,
#         return_sequences=True, 
#         input_shape=(X_train.shape[1], X_train.shape[2])
#         )
#   )
# )
# model.add(Dropout(rate=0.2))
# model.add(LSTM(units =50, activation='relu', return_sequences=True))
# model.add(Dropout(0.15))
# model.add(LSTM(units =50, activation='relu', return_sequences=False))
# model.add(Dense(units=1))
# model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['mean_absolute_error'], name='mean_absolute_error'))
# fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='accuracy'))
# fig.show()

y_pred = model.predict(X_test)
y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

import matplotlib.pyplot as plt

plt.plot(y_test_inv.flatten(), marker='.', label='true')
plt.plot(y_pred_inv.flatten(), 'r', label='predicted')
plt.show()