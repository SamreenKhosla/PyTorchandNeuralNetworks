<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# # Predicting Residential EV Charging Loads using Neural Networks
# 
# - [View Solution Notebook](./solutions.html)
# - [View Project Page](https://www.codecademy.com/)

# In[5]:


# Setup - import basic data libraries
import numpy as np
import pandas as pd


# ## Task Group 1 - Load, Inspect, and Merge Datasets

# ### Task 1
# 
# The file `&#39;datasets/EV charging reports.csv&#39;` contains electric vehicle (EV) charging data. These come from various residential apartment buildings in Norway. The data includes specific user and garage information, plug-in and plug-out times, charging loads, and the dates of the charging sessions.
# 
# Import this CSV file to a pandas DataFrame named `ev_charging_reports`.
# 
# Use the `.head()` method to preview the first five rows.

# In[6]:


ev_charging_reports = pd.read_csv(&#39;datasets/EV charging reports.csv&#39;)
ev_charging_reports.head()


# <details><summary style="display:list-item; font-size:16px; color:blue;">What is the structure of the dataset?</summary>
# 
# - **session_ID** - the unique id for each EV charging session
# - **Garage_ID** - the unique id for the garage of the apartment
# - **User_ID** - the unique id for each user
# - **User_private** - 1.0 indicates private charge point spaces and 0.0 indicates shared charge point spaces
# - **Shared_ID** - the unique id if shared charge point spaces are used
# - **Start_plugin** - the plug-in date and time in the format (day.month.year hour:minute)
# - **Start_plugin_hour** - the plug-in date and time rounded to the start of the hour
# - **End_plugout** - the plug-out date and time in the format (day.month.year hour:minute)
# - **End_plugout_hour** - the start of the hour of the `End_plugout` hour
# - **El_kWh** - the charged energy in kWh (charging loads)
# - **Duration_hours** - the duration of the EV connection time per session
# - **Plugin_category** - the plug-in time categorized by early/late night, morning, afternoon, and evening
# - **Duration_category** - the plug-in duration categorized by 3 hour groups
# - **month_plugin_{month}** - the month of the plug-in session
# - **weekdays_plugin_{day}** - the day of the week of the plug-in session

# ### Task 2
# 
# Import the file `&#39;datasets/Local traffic distribution.csv&#39;` to a pandas DataFrame named `traffic_reports`. This dataset contains the hourly local traffic density counts at 5 nearby traffic locations. 
# 
# Preview the first five rows.

# In[7]:


traffic_reports = pd.read_csv(&#39;datasets/Local traffic distribution.csv&#39;)
traffic_reports.head()


# <details><summary style="display:list-item; font-size:16px; color:blue;">What is the structure of the dataset?</summary>
# 
# - **Date_from** - the starting time in the format (day.month.year hour:minute)
# - **Date_to** - the ending time in the format (day.month.year hour:minute)
# - **Location 1 to 5** - contains the number of vehicles each hour at a specified traffic location.
# 

# ### Task 3
# 
# We&#39;d like to use the traffic data to help our model. The same charging location may charge at different rates depending on the number of cars being charged, so this traffic data might help the model out.
# 
# Merge the `ev_charging_reports` and `traffic_reports` datasets together into a Dataframe named `ev_charging_traffic` using the columns:
# 
# - `Start_plugin_hour` in `ev_charging_reports`
# - `Date_from` in `traffic_reports`

# In[8]:


ev_charging_traffic = pd.merge(
    ev_charging_reports,
    traffic_reports,
    left_on=&#39;Start_plugin_hour&#39;,
    right_on=&#39;Date_from&#39;
)

ev_charging_traffic.head()


# ### Task 4
# 
# Use `.info()` to inspect the merged dataset. Specifically, pay attention to the data types and number of missing values in each column.

# In[9]:


ev_charging_traffic.info()


# <details><summary style="display:list-item; font-size:16px; color:blue;">What do we notice about merged dataset under inspection?</summary>
# 
# We see that there are 39 columns and 6,833 rows in our merged dataset.
# 
# Some notable things we might have to address:
# 
# - We expected columns like `El_kWh` and `Duration_hours` to be floats but they are actually object data types.
# 
# - There are many identifying columns like `session_ID` and `User_ID` that might not be useful for training.

# ## Task Group 2 - Data Cleaning and Preparation

# ### Task 5
# 
# Let&#39;s start by reducing the size of our dataset by dropping columns that won&#39;t be used for training. These include
# - ID columns
# - columns with lots of missing data
# - non-numeric columns (for now, since we haven&#39;t yet covered using non-numeric data in neural networks)
# 
# Drop columns you don&#39;t want to use in training from `ev_charging_traffic`.
# 
# To match our solution, drop the columns
# 
# ```py
# [&#39;session_ID&#39;, &#39;Garage_ID&#39;, &#39;User_ID&#39;, 
#                 &#39;Shared_ID&#39;,
#                 &#39;Plugin_category&#39;,&#39;Duration_category&#39;, 
#                 &#39;Start_plugin&#39;, &#39;Start_plugin_hour&#39;, &#39;End_plugout&#39;, &#39;End_plugout_hour&#39;, 
#                 &#39;Date_from&#39;, &#39;Date_to&#39;]
# ```

# In[10]:


ev_charging_traffic = ev_charging_traffic.drop(columns=[
    &#39;session_ID&#39;, &#39;Garage_ID&#39;, &#39;User_ID&#39;, 
    &#39;Shared_ID&#39;,
    &#39;Plugin_category&#39;,&#39;Duration_category&#39;, 
    &#39;Start_plugin&#39;, &#39;Start_plugin_hour&#39;, 
    &#39;End_plugout&#39;, &#39;End_plugout_hour&#39;, 
    &#39;Date_from&#39;, &#39;Date_to&#39;
])

ev_charging_traffic.head()


# ### Task 6
# 
# Earlier we saw that the `El_kWh` and `Duration_hours` columns were object data types. Upon further inspection, we see that the reason is that the data is following European notation where commas `,` are used as decimals instead of periods.
# 
# Replace `,` with `.` in these three columns.

# In[12]:


ev_charging_traffic[&#39;El_kWh&#39;] = ev_charging_traffic[&#39;El_kWh&#39;].str.replace(&#39;,&#39;, &#39;.&#39;)
ev_charging_traffic[&#39;Duration_hours&#39;] = ev_charging_traffic[&#39;Duration_hours&#39;].str.replace(&#39;,&#39;, &#39;.&#39;)


# ### Task 7
# 
# Next, convert the data types of all the columns of `ev_charging_traffic` to floats.

# In[13]:


ev_charging_traffic = ev_charging_traffic.astype(float)

ev_charging_traffic.info()


# ## Task Group 3 - Train Test Split
# 
# Next, let&#39;s split the dataset into training and testing datasets. 
# 
# The training data will be used to train the model and the testing data will be used to evaluate the model.

# ### Task 8
# 
# First, create two datasets from `ev_charging_traffic`:
# 
# - `X` contains only the input numerical features
# - `y` contains only the target column `El_kWh`

# In[14]:


X = ev_charging_traffic.drop(columns=[&#39;El_kWh&#39;])
y = ev_charging_traffic[&#39;El_kWh&#39;]


# ### Task 9
# 
# Use `sklearn` to split `X` and `y` into training and testing datasets. The training set should use 80% of the data. Set the `random_state` parameter to `2`.

# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=2
)


# ## Task Group 4 - Linear Regression Baseline

# This section is optional, but useful. The idea is to compare our neural network to a basic linear regression. After all, if a basic linear regression works just as well, there&#39;s no need for the neural network!
# 
# If you haven&#39;t done linear regression with scikit-learn before, feel free to use [our solution code](./solutions.html) or to skip ahead.

# ### Task 10
# 
# Use Scikit-learn to train a Linear Regression model using the training data to predict EV charging loads.
# 
# The linear regression will be used as a baseline to compare against the neural network we will train later.

# In[17]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# ### Task 11
# 
# Evaluate the linear regression baseline by calculating the MSE on the testing data. Use `mean_squared_error` from `sklearn.metrics`.
# 
# Save the testing MSE to the variable `test_mse` and print it out.

# In[18]:


from sklearn.metrics import mean_squared_error
y_pred = linear_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(test_mse)


# Looks like our mean squared error is around `131.4` (if you used different columns in your model than we did, you might have a different value). Remember, this is squared error. If we take the square root, we have about `11.5`. One way of interpreting this is to say that the linear regression, on average, is off by `11.5 kWh`.

# ## Task Group 5 - Train a Neural Network Using PyTorch
# 
# Let&#39;s now create a neural network using PyTorch to predict EV charging loads.

# ### Task 12
# 
# First, we&#39;ll need to import the PyTorch library and modules.
# 
# Import the PyTorch library `torch`.
# 
# From `torch`, import `nn` to access built-in code for constructing networks and defining loss functions.
# 
# From `torch`, import `optim` to access built-in optimizer algorithms.

# In[19]:


import torch
from torch import nn
from torch import optim


# ### Task 13
# 
# Before training the neural network, convert the training and testing sets into PyTorch tensors and specify `float` as the data type for the values.

# In[21]:


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test.values, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# ### Task 14
# 
# Next, let&#39;s use `nn.Sequential` to create a neural network.
# 
# First, set a random seed using `torch.manual_seed(42)`.
# 
# Then, create a sequential neural network with the following architecture:
# 
# - input layer with number of nodes equal to the number of training features
# - a first hidden layer with `56` nodes and a ReLU activation
# - a second hidden layer with `26` nodes and a ReLU activation
# - an output layer with `1` node
# 
# Save the network to the variable `model`.

# In[22]:


torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(X_train_tensor.shape[1], 56),  
    nn.ReLU(),
    nn.Linear(56, 26),                    
    nn.ReLU(),
    nn.Linear(26, 1)       
)


# ### Task 15
# 
# Next, let&#39;s define the loss function and optimizer used for training:
# 
# - set the MSE loss function to the variable `loss`
# - set the Adam optimizer to the variable `optimizer` with a learning rate of `0.0007`

# In[23]:


loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0007)


# ### Task 16
# 
# Create a training loop to train our neural network for 3000 epochs.
# 
# Keep track of the training loss by printing out the MSE every 500 epochs.

# In[24]:


epochs = 3000

for epoch in range(epochs):
    # 1) forward pass (predict)
    y_pred = model(X_train_tensor)

    # 2) compute loss
    train_loss = loss(y_pred, y_train_tensor)

    # 3) clear old gradients
    optimizer.zero_grad()

    # 4) backprop (compute new gradients)
    train_loss.backward()

    # 5) update weights
    optimizer.step()

    # print every 500 epochs
    if (epoch + 1) % 500 == 0:
        print(f&#34;Epoch {epoch+1}/{epochs}, Loss: {train_loss.item():.4f}&#34;)


# ### Task 17
# 
# Save the neural network in the `models` directory using the path `models/model.pth`.

# In[25]:


torch.save(model.state_dict(), &#39;models/model.pth&#39;)


# ### Task 18
# 
# Evaluate the neural network on the testing set. 
# 
# Save the testing data loss to the variable `test_loss` and use `.item()` to extract and print out the loss. 

# In[26]:


model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = loss(test_predictions, y_test_tensor)

print(test_loss.item())


# ### Task 19
# 
# We trained this same model for 4500 epochs locally. That model is saved as `models/model4500.pth`. Load this model using PyTorch and evaluate it. How well does the longer-trained model perform?

# In[ ]:


import torch
from torch import nn

model_long = nn.Sequential(
    nn.Linear(X_train_tensor.shape[1], 56),
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)
model_long.load_state_dict(torch.load(&#39;models/model4500.pth&#39;))
model_long.eval()
with torch.no_grad():
    long_predictions = model_long(X_test_tensor)
    long_test_loss = loss(long_predictions, y_test_tensor)

print(long_test_loss.item())


# Pretty cool! The increased training improved our test loss to about `115.2`, a full `12%` improvement on our linear regression baseline. So the nonlinearity introduced by the neural network actually helped us out.

# That&#39;s the end of our project on predicting EV charging loads! Feel free to continue experimenting with this neural network model. 
# 
# Some things you might want to investigate further include:
# - explore different ways to clean and prepare the data
# - we added traffic data, but there&#39;s no guarantee that more data converts to a better model. Test out different sets of input columns.
# - test out different number of nodes in the hidden layers, activation functions, and learning rates
# - train on a larger number of epochs 

# In[ ]:




</details></details></details><script type="text/javascript" src="/relay.js"></script></body></html>
