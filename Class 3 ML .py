#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np #numeric python
import pandas as pd #dataframes
import matplotlib.pylab as plt #visualization


# In[45]:


get_ipython().run_line_magic('pip', 'install --upgrade scikit-learn==0.23.0')


# In[46]:


df = pd.read_csv(r"C:\Users\Lenovo\Downloads\drawndata1.csv") #raw string


# In[47]:


df.head(3) #to print the first 3 rows of your dataframe


# In[48]:


X = df[['x','y']].values   #x variable stores x, y values of the dataframe
y = df['z'] == "a"   #y variables stores z values of the dataframes


# In[63]:


plt.scatter(X[:,0],X[:,1],c=y)  


# In[50]:


from sklearn.preprocessing import StandardScaler
X_new = StandardScaler().fit_transform(X)
plt.scatter(X_new[:,0],X_new[:,1],c=y)


# In[51]:


X = np.random.exponential(10,(1000)) + np.random.normal(0,1,(1000)) #simulating standard scaler
plt.hist(X-np.mean(X),30);


# In[52]:


X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
plt.scatter(X_new[:,0],X_new[:,1],c=y)


# In[53]:


from sklearn.preprocessing import QuantileTransformer
X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
plt.scatter(X_new[:,0],X_new[:,1],c=y)


# In[54]:


get_ipython().run_line_magic('pip', 'install --upgrade scikit-learn==0.23.0')


# In[ ]:




