#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile,mutual_info_classif
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv("E:\liu\lab\pk\Customers.csv")


# In[6]:


df.head()


# In[7]:


mean_of_Income=np.mean(df['Annual Income ($)'])
print('mean_of_Income=',mean_of_Income)


# In[8]:


maximum=max(df['Annual Income ($)'])
minimum=min(df['Annual Income ($)'])
l=maximum-minimum
new_Income=(df['Annual Income ($)']-mean_of_Income)/l
print(new_Income)


# In[87]:


X,y=load_digits(return_X_y=True)
X.shape


# In[88]:


X_new=SelectPercentile(mutual_info_classif,percentile=10).fit_transform(X,y)
X_new.shape


# In[89]:


plt.figure(figsize=(8,4))
plt.scatter(df['Age'],df['Annual Income ($)'])
plt.xlabel='Age'
plt.ylabel='Annual Income ($)'
plt.show()


# In[ ]:




