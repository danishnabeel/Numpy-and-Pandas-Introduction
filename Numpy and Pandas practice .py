#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#creating list

l = [1,2,3]


# In[3]:


np.array(l)


# In[4]:


my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix


# In[5]:


np.array(my_matrix)


# Built in methods

# In[6]:


np.arange(0,10)


# In[7]:


np.arange(0,11,2)


# In[8]:


#creating identity matrix
np.eye(3)


# ## Random 
# 
# Numpy also has lots of ways to create random number arrays:
# 
# ### rand
# Create an array of the given shape and populate it with
# random samples from a uniform distribution
# over ``[0, 1)``.

# In[9]:


np.random.rand(2)


# 
# 

# ## SERIES

# In[10]:


import numpy as np
import pandas as pd


# ### Creating a Series
# 
# You can convert a list,numpy array, or dictionary to a Series:

# In[20]:


labels = ['a','b','c']
l = [10,20,30]
array = np.array([10,20,30])
d = {'a':10,'b':20,'c':30}


# In[21]:


pd.Series(data=l)


# In[22]:


pd.Series(l,labels)


# In[23]:


pd.Series(array,labels)


# In[25]:


pd.Series(d)


# In[26]:


ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])                                   


# In[27]:


ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])                                   


# In[28]:


#adding based on index
ser1 + ser2


# In[29]:


ser1['USA']


# # DataFrames

# In[30]:


from numpy.random import randn


# In[33]:


df = pd.DataFrame(randn(5,4), index = ['a','b','c','d','e'], columns= ['q','w','e','t'])


# In[35]:


df


# In[36]:


# Pass a list of column names
df[['q','t']]


# In[38]:


#creating new column
df['new'] = df['e'] + df['t']


# In[39]:


df


# In[40]:


df.drop('new', axis=1, inplace=True) #dropping newly added column


# In[41]:


df


# In[42]:


df.loc[['a','b'], ['q','w']] #subset of rows


# In[49]:



df[df['e']>0][['e','t']] #conditional statement


# In[52]:


df[(df['w']>0) & (df['e'] > 1)]


# # Missing Data
# 
# Dealing with missing values

# In[53]:


df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})


# In[54]:


df


# In[55]:


df.dropna()


# In[56]:


df.dropna(axis=1)


# In[63]:


df.dropna(thresh=2)


# In[64]:


df.fillna('value')


# In[66]:


df


# In[67]:


df['A'].fillna(value=df['A'].mean()) #filling missing value with average of the column


# # Groupby
# 
# The groupby method allows you to group rows of data together and call aggregate functions

# In[68]:



# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}


# In[69]:


df = pd.DataFrame(data)


# In[70]:


df


# In[72]:


df.groupby('Company').mean()


# In[73]:


df.groupby('Company').min()


# In[75]:


df.groupby('Company').describe().transpose()


# In[76]:


df.groupby('Company').describe().transpose()['FB']


# In[ ]:




