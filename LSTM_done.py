#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas_datareader')
import matplotlib


# In[2]:


import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import numpy


# In[3]:


aapl = yf.Ticker('aapl')
df = aapl.history(period='max')
df.to_csv('AAPL.csv')
df=pd.read_csv('AAPL.csv')
df1 = df[['Close','High','Low','Open','Volume']]

val_final = df1[-100:][['Close']]
val_final = val_final.reset_index()
val_final = val_final['Close']
print(val_final)


# In[4]:


def get_technical_indicators(dataset):
    # 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
    
    # MACD
    ema_26 = pd.DataFrame(data = dataset['Close'])
    ema_12 = pd.DataFrame(data = dataset['Close'])
    dataset['26ema'] = ema_26.ewm(span=26).mean()
    dataset['12ema'] = ema_12.ewm(span=12).mean()

    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
    
    # log momentum
    dataset['momentum'] = dataset['Close']/100-1
    
    return dataset


# In[5]:


get_technical_indicators(df1)
print(df1)


# In[6]:


import matplotlib.pyplot as plt
import numpy as np
df1 = df1[-4000:]
plt.plot(df1['Close'])


# In[7]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

df1=scaler.fit_transform(df1)
print(df1)
val_data = df1[-200:]
df1 = df1[:-200]


# In[8]:


training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:]
training_size,test_size


# In[9]:


print(train_data)


# In[10]:


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), :]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)
 
time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
x_val, y_val = create_dataset(val_data,time_step)


# In[ ]:





# In[11]:


print(x_train.shape), print(y_train.shape)
print(x_test.shape), print(y_test.shape)


# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model=Sequential()
model.add(LSTM(50,activation='ReLU',return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(50,activation='ReLU',return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=250,batch_size=64,verbose=1)


# In[17]:


import tensorflow as tf
tf.__version__
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)
val_predict = model.predict(x_val)

pred_train = pd.concat([pd.DataFrame(train_predict),pd.DataFrame(train_data[101:,1:])],axis=1)
pred_train=scaler.inverse_transform(pred_train)

pred_test = pd.concat([pd.DataFrame(test_predict),pd.DataFrame(test_data[101:,1:])],axis=1)
pred_test=scaler.inverse_transform(pred_test)

pred_val = pd.concat([pd.DataFrame(val_predict),pd.DataFrame(val_data[101:,1:])],axis=1)
pred_val=scaler.inverse_transform(pred_val)

val_predict = pred_val[:,0]
#test_predict = pred_test[:,0]
train_predict = pred_train[:,0]

print(val_predict.shape,val_final.shape)
plt.plot(test_predict)
plt.plot(y_test)


# In[16]:


plt.plot(val_predict)
plt.plot(val_final)


# In[18]:


import math
from sklearn.metrics import mean_squared_error
print(math.sqrt(mean_squared_error(y_train,train_predict)))
print(math.sqrt(mean_squared_error(y_test,test_predict)))
print(math.sqrt(mean_squared_error(val_final,val_predict)))


# In[ ]:




