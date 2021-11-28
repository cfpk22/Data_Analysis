#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


columns_name = ['user_id', 'item_id', 'rating', 'timestamp']


# In[5]:


df = pd.read_csv('u.data', sep='\t', names=columns_name)


# In[6]:


df.head()


# In[7]:


movie_titles = pd.read_csv('Movie_Id_Titles')


# 

# In[8]:


movie_titles.head()


# In[9]:


df = pd.merge(df,movie_titles,on='item_id')


# In[10]:


df.head()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[14]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[15]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[16]:


ratings.head()


# In[17]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[18]:


ratings.head()


# In[19]:


ratings['num of ratings'].hist(bins=70)


# In[20]:


ratings['rating'].hist(bins=70)


# In[22]:


import scipy.stats as stats
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5).annotate(stats.pearsonr)


# In[24]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')


# 

# In[25]:


moviemat


# In[26]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# In[27]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']


# In[28]:


starwars_user_ratings.head()


# In[30]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[31]:


similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# In[34]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)


# In[35]:


corr_starwars.head()


# In[36]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[37]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])


# In[38]:


corr_starwars.head()


# In[39]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',)


# In[40]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])


# In[47]:


corr_liarliar.dropna(inplace=True)


# In[48]:


corr_liarliar = corr_liarliar.join(ratings['num of ratings'])


# In[53]:


corr_liarliar[corr_liarliar['num of ratings']>20].sort_values('Correlation', ascending=False).head()


# In[ ]:




