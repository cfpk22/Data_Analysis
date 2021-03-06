#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Choropleth Maps Exercise 
# 
# Welcome to the Choropleth Maps Exercise! In this exercise we will give you some simple datasets and ask you to create Choropleth Maps from them. Due to the Nature of Plotly we can't show you examples
# 
# [Full Documentation Reference](https://plot.ly/python/reference/#choropleth)
# 
# ## Plotly Imports

# In[1]:


import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True) 


# ** Import pandas and read the csv file: 2014_World_Power_Consumption**

# In[2]:


import pandas as pd


# In[5]:


df = pd.read_csv('2014_World_Power_Consumption')


# ** Check the head of the DataFrame. **

# In[4]:


df.head()


# ** Referencing the lecture notes, create a Choropleth Plot of the Power Consumption for Countries using the data and layout dictionary. **

# In[13]:


data = dict(type='choropleth',
            colorscale = 'ylorbr',
            locations = df['Country'],
            z = df['Power Consumption KWH'],
            reversescale = True,
            locationmode = 'country names',
            text = df['Text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"KWH"}
            ) 
layout = dict(
    title = '2014 World Power Consumption',
    geo = dict(
        showframe = False,
        projection = {'type':'mercator'}
    )
)


# In[14]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# ## USA Choropleth
# 
# ** Import the 2012_Election_Data csv file using pandas. **

# In[15]:


df2 = pd.read_csv('2012_Election_Data')


# ** Check the head of the DataFrame. **

# In[16]:


df2.head()


# ** Now create a plot that displays the Voting-Age Population (VAP) per state. If you later want to play around with other columns, make sure you consider their data type. VAP has already been transformed to a float for you. **

# In[120]:


data = dict(type='choropleth',
            colorscale = 'ylorbr',
            locations = df['State Abv'],
            z = df['Voting-Age Population (VAP)'],
            reversescale=True,
            locationmode = 'USA-states',
            text = df['State'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Millions USD"}
            ) 


# In[121]:


layout = dict(title = '2011 US Agriculture Exports by State',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )


# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# # Great Job!
