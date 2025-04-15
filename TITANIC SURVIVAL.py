

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


df=pd.read_csv("titanic.csv")
df.head(10)



df.shape



df.describe()



df['Survived'].value_counts()


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




