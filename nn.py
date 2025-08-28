import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
import keras.api.metrics
from sklearn.metrics import r2_score

data = pd.read_csv('Battery_RUL.csv')
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())

# Detect columns with fewer than 10 unique values
for i in data.columns.values:
    if len(data[i].value_counts().values) < 10:
        print(data[i].value_counts())

# Outlier removal using z-scores
out = []
for i in data.columns.values:
    data['z_scores'] = (data[i] - data[i].mean()) / data[i].std()
    outlier = np.abs(data['z_scores'] > 3).sum()
    if outlier > 3:
        out.append(i)

print(len(data))
thresh = 3
for i in out:
    upper = data[i].mean() + thresh * data[i].std()
    lower = data[i].mean() - thresh * data[i].std()
    data = data[(data[i] > lower) & (data[i] < upper)]

print(len(data))

# Correlation with RUL
corr = data.corr()['RUL']
corr = corr.drop(['RUL', 'z_scores'])
x_cols = [i for i in corr.index if corr[i] > 0]
x = data[x_cols]
y = data['RUL']

x_train,x_test,y_train,y_test=train_test_split(x,y)

models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.linear))
models.add(Dense(units=x.shape[1],activation=keras.activations.linear))
models.add(Dense(units=1,activation=keras.activations.linear))
models.compile(optimizer='adam',loss=keras.losses.mean_squared_error,metrics=['mse'])
models.fit(x_train,y_train,epochs=300,batch_size=150,validation_data=(x_test,y_test))
models.save('rlu.h5')