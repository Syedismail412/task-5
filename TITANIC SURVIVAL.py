#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("titanic.csv")
df.head(10)


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df['Survived'].value_counts()


# In[6]:


#visualise the count of survivals wrt pclass
sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[7]:


df['Sex']


# In[8]:


#viualise the count of survivals wrt gender
sns.countplot(x=df['Sex'],hue=df['Survived'])


# In[9]:


df.groupby('Sex')[['Survived']].mean()


# In[10]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

df['Sex']=labelencoder.fit_transform(df['Sex'])
df.head()


# In[ ]:




