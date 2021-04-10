#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Seaborn Exercises
# 
# Time to practice your new seaborn skills! Try to recreate the plots below (don't worry about color schemes, just the plot itself.

# ## The Data
# 
# We will be working with a famous titanic data set for these exercises. Later on in the Machine Learning section of the course, we will revisit this data, and use it to predict survival rates of passengers. For now, we'll just focus on the visualization of the data with seaborn:

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sns.set_style('whitegrid')


# In[3]:


titanic = sns.load_dataset('titanic')


# In[5]:


titanic.head()


# # Exercises
# 
# ** Recreate the plots below using the titanic dataframe. There are very few hints since most of the plots can be done with just one or two lines of code and a hint would basically give away the solution. Keep careful attention to the x and y labels for hints.**
# 
# ** *Note! In order to not lose the plot image, make sure you don't code in the cell that is directly above the plot, there is an extra cell above that one which won't overwrite that plot!* **

# In[7]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
sns.jointplot(x='fare',y='age',data=titanic)


# In[41]:





# In[15]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
sns.distplot(titanic['fare'],kde=False, bins=30, color='red')


# In[44]:





# In[19]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
sns.boxplot(x='class',y='age', data=titanic, palette='rainbow')


# In[45]:





# In[ ]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!


# In[20]:


sns.swarmplot(x='class',y='age', data=titanic)


# In[21]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
sns.countplot(titanic['sex'])


# In[47]:





# In[25]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
tc = titanic.corr()
sns.heatmap(tc, cmap='coolwarm')
plt.title('titanic.corr()')


# In[48]:





# In[27]:


# CODE HERE
# REPLICATE EXERCISE PLOT IMAGE BELOW
# BE CAREFUL NOT TO OVERWRITE CELL BELOW
# THAT WOULD REMOVE THE EXERCISE PLOT IMAGE!
sns.distplot(titanic['age'], hue='sex')


# In[49]:





# # Great Job!
# 
# ### That is it for now! We'll see a lot more of seaborn practice problems in the machine learning section!
