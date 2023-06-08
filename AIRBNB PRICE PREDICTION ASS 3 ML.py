#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[56]:


df = pd.read_csv("airbnb_listing_train.csv")


# In[57]:


df


# In[58]:


df.info()


# In[59]:


df.isnull().sum()


# In[60]:


df.shape


# In[64]:


X = df.drop('availability_365', axis = 1)
y = df['price']


# In[65]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[66]:


from sklearn.linear_model import LinearRegression
lr  = LinearRegression()
lr


# In[ ]:




