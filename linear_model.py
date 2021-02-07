import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

df = quandl.get("BSE/BOM532215")
print(df.head())

df = df[['Close']]
print(df.head())

forecast_out = 60    #Number of days in the future
df['Prediction'] = df[['Close']].shift(-forecast_out)
print(df.tail())

X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)

y = np.array(df['Prediction'])
y = y[:-forecast_out]
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

svr_rbf = SVR(kernel='rbf', C= 1e3, gamma = 0.1)
svr_rbf.fit(x_train, y_train)

svm_confidence = svr_rbf.score(x_test,y_test)
print("svm confidence: ", svm_confidence)

lr = LinearRegression()
lr.fit(x_train, y_train)

lr_confidence = lr.score(x_test,y_test)
print("lr confidence: ", lr_confidence)

x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

plt.figure(figsize=(14,5))
plt.plot(y_test[-forecast_out:], color = 'red', label = 'Real AXIS Price')
plt.plot(lr_prediction, color = 'blue', label = 'Predicted AXIS Price')
plt.title('AXIS Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AXIS Stock Price')
plt.legend()
plt.show()

plt.figure(figsize=(14,5))
plt.plot(y_test[-forecast_out:], color = 'red', label = 'Real AXIS Price')
plt.plot(svm_prediction, color = 'blue', label = 'Predicted AXIS Price')
plt.title('AXIS Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AXIS Stock Price')
plt.legend()
plt.show()