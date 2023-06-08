#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[9]:


df = pd.read_csv("HR_comma_sep.csv.txt")


# In[10]:


df


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


sns.countplot(x='left',data=df)
plt.title('Employee Exit Distribution')
plt.show()


# In[14]:


df = pd.get_dummies(df,columns=['sales','salary'],drop_first = True)


# In[15]:


df.fillna(df.mean(),inplace=True)


# In[16]:


df


# In[17]:


x = df.drop('left',axis =1)
y = df['left']


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[19]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train_scaled, y_train)


# In[20]:


y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)


# In[21]:


print('Accuracy:', accuracy)
print('Confusion Matrix:')
print(confusion_mat)


# In[22]:


feature_importances = model.feature_importances_
feature_names = x.columns


# In[24]:


plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




