#Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Step 2: Load dataset
dataset = pd.read_csv('D:\esa\PROJECTS\StockPrice_Prediction\Stock_Price_dataset.csv')
dataset.head()
dataset.info()
dataset.shape


#Step 3: Covert Date to datetime format
dataset["Date"] = pd.to_datetime(dataset.Date, format="%Y-%m-%d")


#Step 4: Use Open Stock Price Column to Train Model
training_set = dataset.iloc[0:949,1:2].values
training_set.shape

#Actual Values
actual_stock_price = dataset.iloc[0:949,1:2].values
actual_stock_price.shape

testing_set = dataset.iloc[949:,1:2].values
testing_set.shape


#Step 5: Normalize the Dataset (Common scale to compare data sets with very different values.)
from sklearn.preprocessing import MinMaxScaler

#Calculate the range of the data set
scaler = MinMaxScaler(feature_range = (0,1))

scaled_training_set = scaler.fit_transform(training_set)

scaled_training_set.shape


#Step 6: Create X_train and Y_train
X_train = []
Y_train = []
for i in range(60, len(training_set)):
    X_train.append(scaled_training_set[i-60:i, 0])
    Y_train.append(scaled_training_set[i, 0])
X_train = np.array(X_train)
#X_train has (949, 60)
Y_train = np.array(Y_train)
#Y_train has (949,)

print(X_train.shape)
print(Y_train.shape)


#Step 7: Reshape Data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


#Step 8: Importing Libraries and adding different layers to LSTM
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))


#Step 9: Fitting the Model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)


#Step 10: Preparing the Input
dataset_total = dataset['Open']
inputs = dataset_total[len(dataset_total) - len(testing_set) - 60:].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
inputs.shape

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape


#Step 11: Predicting the values
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


#Step 12: Plotting Actual and Predicted Prices
training_set = dataset[:949]
testing_set = dataset[949:]
testing_set['Predictions'] = predicted_stock_price
plt.plot(training_set["Close"])
plt.plot(testing_set[['Close',"Predictions"]])
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')