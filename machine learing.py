#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn import linear_model


# In[5]:


data = pd.read_csv('C:/Jupyter_working_path/1/data103.csv', sep=',', encoding='latin-1')
data = data.drop(columns=['Date'])
data = np.array(data)


# In[3]:


train_index = 2400
val_index = 2650


# In[4]:


KOSPI = 0	
Volume = 1
KOSPI_1W = 2
KOSPI_1M = 3
KOSPI_3M = 4
KOSPI_6M = 5
KOSPI_1Y = 6
Vol_5d = 7
Vol_20d = 8
Vol_60d = 9
Vol_120d = 10
Vol_1Y = 11
Foreign_s = 12
Short = 13
Short2 = 14
Short2_change = 15
Inst = 16
Individual_p = 17
Foreign_p = 18
Interest = 19
Exchange = 20
SP500 = 21
Nasdaq = 22
Semi = 23
WTI = 24
Copper = 25
kospi_52wa = 43

summary = []


# In[5]:


for i in range(1, 10):
    
    delay = i
    
    # Creating test datasets
    train_targets = data[delay:train_index, KOSPI]
    train_data = data[:train_index-delay]
    val_targets = data[train_index+delay:val_index, KOSPI]
    val_data = data[train_index:val_index-delay]
    test_targets = data[val_index+delay:, KOSPI]
    test_data = data[val_index:-delay]
    
    def evaluate_naive_method():
        naive_preds = np.zeros(len(test_targets))
        for i in range(len(test_targets)):
            reg = linear_model.LinearRegression()
            x = data[:val_index+i, Volume:kospi_52wa+1]
            y = data[delay:val_index+i+delay, KOSPI]
            reg.fit(x, y)
            naive_preds[i] = reg.intercept_ + np.dot(reg.coef_, data[val_index+i+delay-1, Volume:kospi_52wa+1])
        return np.mean((test_targets - naive_preds)**2)
    
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [EarlyStopping(monitor='val_mean_squared_error', patience=80), 
                 ModelCheckpoint(('C:\Jupyter_working_path\1\model-DNN.h5'), save_best_only=True, save_weights_only=False)]


# In[6]:


# Deep Neural Network
from keras import models
from keras import layers
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import PReLU   


# In[7]:


model = models.Sequential()
model.add(layers.Dense(43,
                           activation='sigmoid', 
                           input_shape=(train_data.shape[-1],)))
model.add(layers.Dense(64))
model.add(PReLU())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128))
model.add(PReLU())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64))
model.add(PReLU())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(32))
model.add(PReLU())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(16))
model.add(PReLU())
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(8))
model.add(PReLU())
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(1))


# In[8]:


model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])


# In[9]:


history = model.fit(train_data, train_targets,
              epochs = 300, 
              batch_size=256,
              callbacks=callbacks,
              validation_data=(val_data, val_targets),
              verbose=True)


# In[10]:


#Predictions
from keras.models import load_model
model = load_model('C:\Jupyter_working_path\1\model-DNN.h5')
predictions = model.predict(test_data)
naive_rmse = evaluate_naive_method()
DNN_rmse = np.mean((test_targets - predictions)**2)
summary.append([i, naive_rmse, DNN_rmse])
print('MSE (Naive regression) :', naive_rmse)
print('MSE (Deep learning:DNN):', DNN_rmse)


# In[11]:


final_data = data[-1:]
predict = model.predict(final_data)
print('\nNext +',delay,'day prediction: ', predict, '\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




