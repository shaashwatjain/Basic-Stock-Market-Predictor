import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("INFY.csv", date_parser=True)
print(data.tail())
data_training = data[data['Date']<'2019-10-01'].copy()
print(data_training)
data_test = data[data['Date']>='2019-10-01'].copy()
print(data_test)

data_training = data_training.drop({'Date'},axis = 1)
scalar = MinMaxScaler()
training_data = scalar.fit_transform(data_training)

X_train = []
y_train = []

for i in range(30,training_data.shape[0]):
    X_train.append(training_data[i-30:i])
    y_train.append(training_data[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape, y_train.shape)

from keras import Sequential
from keras.layers import Dense, LSTM, Dropout

regression = Sequential()
regression.add(LSTM(units= 60, activation = "relu", return_sequences = True, input_shape = (X_train.shape[1],5)))
regression.add(Dropout(0.2))

regression.add(LSTM(units= 60, activation = "relu", return_sequences = True))
regression.add(Dropout(0.2))

regression.add(LSTM(units= 80, activation = "relu", return_sequences = True))
regression.add(Dropout(0.2))

regression.add(LSTM(units= 120, activation = "relu"))
regression.add(Dropout(0.2))

regression.add(Dense(units=1))
regression.summary()

regression.compile(optimizer="adam", loss = "mean_squared_error")
regression.fit(X_train,y_train,epochs = 5 ,batch_size=128)

past_60_days = data_training.tail(30)
df = past_60_days.append(data_test, ignore_index = True)
df = df.drop({'Date'}, axis = 1)
inputs = scalar.transform(df)

X_test = []
y_test = []

for i in range(30, inputs.shape[0]):
    X_test.append(inputs[i-30:i])
    y_test.append(inputs[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)
y_pred = regression.predict(X_test)
print(y_pred)

scale = 1/scalar.scale_[0]

y_pred = y_pred*scale
y_test = y_test*scale

plt.figure(figsize=(14,5))
plt.plot(y_train, color = 'red', label = 'Real Price')
plt.title('Stock Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(y_pred, color = 'blue', label = 'Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
