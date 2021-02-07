import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('AXISBANK.csv', date_parser = True)


data_training = data[data['Date']<'2019-01-01'].copy()
data_test = data[data['Date']>='2019-01-01'].copy()

data_training = data_training.drop(['Date'], axis = 1)

scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)

X_train = []
y_train = []

for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

from keras import Sequential
from keras.layers import Dense, LSTM, Dropout

regressior = Sequential()

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 120, activation = 'relu'))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))

regressior.summary()

regressior.compile(optimizer='adam', loss = 'mean_squared_error')

regressior.fit(X_train, y_train, epochs=1, batch_size=32)

past_60_days = data_training.tail(60)
df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date'], axis = 1)

inputs = scaler.transform(df)


X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
y_pred = regressior.predict(X_test)
scaler.scale_
scale = 1/8.18605127e-04
y_pred = y_pred*scale
y_test = y_test*scale

plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
