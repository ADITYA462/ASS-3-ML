#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[11]:


df = pd.read_csv("mobile_price_data.csv")


# In[12]:


df.describe()


# In[13]:


df.info()


# In[14]:


df.head


# In[15]:


df.shape


# In[16]:


df.dtypes


# In[17]:


df.isna().sum()


# In[18]:


df['mob_weight'] = df['mob_weight'].str.replace('g','')


# In[19]:


df['mob_height']=df['mob_height'].str.replace('mm','')
df['mob_depth']=df['mob_depth'].str.replace('mm','')
df['mob_width']=df['mob_width'].str.replace('mm','')
df['battery_power']=df['battery_power'].str.replace('mAh','')


# In[20]:


df['mp_speed']=df['mp_speed'].str.replace('GHz','')
df['int_memory']=df['int_memory'].str.replace('GB','')
df['ram']=df['ram'].str.replace('GB','')


# In[21]:


df['bluetooth'].value_counts()


# In[22]:


df.drop('bluetooth',axis=1,inplace=True)


# In[23]:


df.head()


# In[24]:


df['num_cores'].value_counts()


# In[25]:


df['num_cores']=df['num_cores'].map({'Octa Core':8,'Quad Core':4,'Single Core':1})


# In[26]:


df['dual_sim'].value_counts()


# In[27]:


df.drop('dual_sim',axis=1,inplace=True)


# In[28]:


df['os'].unique()


# In[29]:


df['mobile_color'].unique()


# In[30]:


df[['dummy1','dummy2','mobile_color']]=df['mobile_color'].str.rpartition(' ')


# In[31]:


df.drop(['dummy1','dummy2'],axis=1,inplace=True)


# In[32]:


df['mobile_color'].value_counts()


# In[33]:


df['mobile_color']=df['mobile_color'].replace({'Greener':'Green','white':'White','gold':'Gold','Gray':'Grey'})


# In[34]:


df['mobile_color'].value_counts()


# In[35]:


df[['mobile_name','dummy1','dummy2']]=df['mobile_name'].str.partition(' ')


# In[36]:


df.drop(['dummy1','dummy2'],axis=1,inplace=True)


# In[37]:


df['mobile_name'].value_counts()


# In[38]:


df['network'].head()


# In[39]:


df['network']=df['network'].str.replace(' ','')


# In[40]:


df['network']=df['network'].apply(lambda x: sorted(x.split(',')))


# In[41]:


df['network'].value_counts()


# In[42]:


from sklearn.preprocessing import MultiLabelBinarizer

mlb=MultiLabelBinarizer()
dg=pd.DataFrame(mlb.fit_transform(df['network']),columns=mlb.classes_,index=df.index)


# In[43]:


dg


# In[44]:


df['p_cam']


# In[45]:


df['p_cam_max']=[x[0:2].replace('M','') for x in df['p_cam']]
df['p_cam_count'] = [x.count('MP') for x in df['p_cam']]


# In[46]:


df['f_cam']


# In[47]:


df['f_cam_max']=[x[0:2].replace('M','') for x in df['f_cam']]
df['f_cam_count'] = [x.count('MP') for x in df['f_cam']]


# In[48]:


df.drop(['f_cam','p_cam'],axis=1,inplace=True)


# In[49]:


df.head()


# In[50]:


df_mobile_name=pd.get_dummies(df['mobile_name'],dtype=int)


# In[51]:


df_mobile_name


# In[52]:


df_mobile_color=pd.get_dummies(df['mobile_color'],dtype=int)


# In[53]:


df_mobile_color


# In[54]:


df=pd.concat([df,df_mobile_name,df_mobile_color],axis=1)


# In[55]:


df


# In[56]:


df.drop(['mobile_name','mobile_color'],axis=1,inplace=True)


# In[57]:


df.dtypes


# In[58]:


df.select_dtypes(include='number').columns


# In[59]:


df.select_dtypes(include='object').columns


# In[60]:


int_col_list=['mobile_price', 'os',  'int_memory', 'ram',
       'battery_power',
       'p_cam_max', 'f_cam_max']


# In[61]:


float_col_list=['disp_size','mp_speed', 'mob_height', 'mob_width', 'mob_depth', 'mob_weight']


# In[63]:


df.dtypes.value_counts()


# In[ ]:




