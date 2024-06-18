#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


df = pd.read_csv("creditcard.csv")


# In[3]:


df.info()


# In[4]:


#top 5 rows in dataset
df.tail()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()/len(df)*100


# In[7]:


df.columns


# In[8]:


len(df.columns)


# In[9]:


len(df)


# In[10]:


df.head(15)


# In[11]:


df.plot(kind='box')
plt.show()


# In[12]:


feature_with_na = [i for i in df.columns if df[i].isnull().sum()> 1]


# In[13]:


feature_with_na


# In[14]:


def credit(df):
    o = pd.DataFrame()
    for i in df.columns:
        mean_vale = np.mean(df[i])
        std_vale = np.std(df[i])
        z = (df[i]- mean_vale)/std_vale
        o[i] =df[i][(z>3)|(z<-3)]
    return o
        


# In[15]:


outliers = credit(df)


# In[16]:


outliers


# In[17]:


def credit_without_outlier(df):
    cleaned = df.copy()
    for i in df.columns:
        m = np.mean(df[i])
        s = np.std(df[i])
        z = (df[i]- m)/s
        cleaned = cleaned[(z<3) & (z>-3)]
    return cleaned


# In[18]:


cleaned = credit_without_outlier(df)


# In[19]:


cleaned


# In[20]:


outliers = credit(cleaned)


# In[21]:


outliers


# In[22]:


cleaned


# In[23]:


class_count = df['Class'].value_counts()


# In[24]:


class_count


# In[25]:


import matplotlib.pyplot as plt


# In[26]:


class_count.plot(kind = 'bar')


# In[34]:


class0 = class_count[0]


# In[35]:


class1 = class_count[1]


# In[43]:


from sklearn.utils import resample


# In[28]:


from imblearn.over_sampling import SMOTE


# In[61]:


df['Class'].value_counts()


# In[67]:


majority = df[df['Class'] == 0]


# In[36]:


class0, class1


# In[68]:


minority = df[df['Class'] == 1]


# In[51]:


len(minority)


# In[69]:


oversampled_majority = resample(majority,
                                replace = True,
                                n_samples = len(minority),
                                random_state = 42
                               )


# In[70]:


oversampled_data = pd.concat([minority,oversampled_majority ])


# In[73]:


oversampled_data.shape


# In[75]:


oversampled_data['Class'].value_counts()


# In[78]:


X_resam = oversampled_data.drop('Class', axis = 1)
Y_resam = oversampled_data['Class']


# In[89]:


import seaborn as sns


# In[82]:


var = VarianceThreshold(threshold =0)


# In[90]:


corr = X_resam.corr()


# In[93]:


corr.shape


# In[94]:


X_resam.shape


# In[95]:


corr


# In[96]:


from sklearn.model_selection import train_test_split


# In[100]:


X_train, X_test, y_train, y_test = train_test_split(X_resam, Y_resam, test_size = 0.2)


# In[101]:


from sklearn.linear_model import LogisticRegression


# In[102]:


lr = LogisticRegression()


# In[103]:


lr.fit(X_train , y_train)


# In[105]:


from sklearn.metrics import accuracy_score


# In[106]:


y_pred = lr.predict(X_test)


# In[107]:


acc = accuracy_score(y_test, y_pred)


# In[108]:


acc


# In[109]:


from sklearn.metrics import confusion_matrix


# In[110]:


cm = confusion_matrix(y_test, y_pred)


# In[111]:


cm


# In[ ]:




