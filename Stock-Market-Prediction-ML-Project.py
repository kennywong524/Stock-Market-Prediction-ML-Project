#!/usr/bin/env python
# coding: utf-8

# 

# # Finance Data Project
# 
# In this data project we will focus on exploratory data analysis of stock prices.
# ____
# We'll focus on bank stocks and see how they progressed throughout the [financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%9308) all the way to early 2016.

# ## Get the Data
# 
# In this section we will use pandas to directly read data from Google finance!
# 

# In[1]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data
# 
# We will get stock information for the following banks:
# *  Bank of America
# * CitiGroup
# * Goldman Sachs
# * JPMorgan Chase
# * Morgan Stanley
# * Wells Fargo
# 
# ** Figure out how to get the stock data from Jan 1st 2006 to Jan 1st 2016 for each of these banks. Set each bank to be a separate dataframe, with the variable name for that bank being its ticker symbol.**
# 1. Use datetime to set start and end datetime objects.
# 2. Figure out the ticker symbol for each bank.
# 2. Figure out how to use datareader to grab info on the stock.
# 

# In[2]:


start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)


# In[3]:


# Bank of America
BAC = data.DataReader("BAC", 'google', start, end)

# CitiGroup
C = data.DataReader("C", 'google', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'google', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'google', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'google', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'google', start, end)


# In[4]:


# Could also do this for a Panel Object
df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'google', start, end)


# ** Create a list of the ticker symbols (as strings) in alphabetical order**

# In[5]:


tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']


# In[6]:


bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)


# In[7]:


bank_stocks.columns.names = ['Bank Ticker','Stock Info']


# In[8]:


bank_stocks.head()


# # EDA
# 
# Let's explore the data a bit!
# 
# **max Close price for each bank's stock throughout the time period?**

# In[9]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()


# ** Create a new empty DataFrame called returns to compute the return for each stock**
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$

# In[10]:


returns = pd.DataFrame()


# In[11]:


for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()


# **Create a pairplot using seaborn of the returns dataframe**

# In[13]:


#returns[1:]
import seaborn as sns
sns.pairplot(returns[1:])


# Background on [Citigroup's Stock Crash available here.](https://en.wikipedia.org/wiki/Citigroup#November_2008.2C_Collapse_.26_US_Government_Intervention_.28part_of_the_Global_Financial_Crisis.29) 

# ** Dates each bank stock had the best and worst single day returns. 4 of the banks share the same day for the worst drop**

# In[14]:


# Worst Drop (4 of them on Inauguration day)
returns.idxmin()


# **Citigroup's largest drop and biggest gain were very close to one another, they had a stock split **

# [Citigroup had a stock split.](https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=citigroup+stock+2011+may)

# In[15]:


# Best Single Day Gain
# citigroup stock split in May 2011, but also JPM day after inauguration.
returns.idxmax()


# ** Take a look at the standard deviation of the returns, we can see which stock was the riskiest for 2015. 

# In[16]:


returns.std() # Citigroup riskiest


# In[17]:


returns.ix['2015-01-01':'2015-12-31'].std() # Very similar risk profiles, but Morgan Stanley or BofA


# ** Create a distplot using seaborn of the 2015 returns for Morgan Stanley **

# In[18]:


sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)


# ** Create a distplot using seaborn of the 2008 returns for CitiGroup **

# In[19]:


sns.distplot(returns.ix['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)


# ____
# # More Visualization
# 
# ### Imports

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly
import cufflinks as cf
cf.go_offline()


# ** A line plot showing Close price for each bank for the entire index of time.

# In[21]:


for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()


# In[22]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()


# In[23]:


# plotly
bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()


# ## Moving Averages
# 
# Let's analyze the moving averages for these stocks in the year 2008. 
# 
# ** THe rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008**

# In[24]:


plt.figure(figsize=(12,6))
BAC['Close'].ix['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].ix['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()


# **A heatmap of the correlation between the stocks Close Price.**

# In[25]:


sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# Use seaborn's clustermap to cluster the correlations together:

# In[26]:


sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)

