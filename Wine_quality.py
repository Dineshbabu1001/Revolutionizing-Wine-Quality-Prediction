#!/usr/bin/env python
# coding: utf-8

# # ***Wine*** ***Quality*** ***Prediction***

# ## Importing the dependencies 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ## Data Collection 

# In[ ]:


wine_dataset = pd.read_csv('/content/winequality-red.csv')


# 

# In[ ]:


#no.s of rows and columns
wine_dataset.shape


# In[ ]:


#first five rows of dataset
wine_dataset.head()


# In[ ]:


#last five rows of dataset
wine_dataset.tail()


# In[ ]:


#checking for missing values 
wine_dataset.isnull().sum()


# # Data Analysis and Visulization

# In[ ]:


#measures of stats of datasets
wine_dataset.describe()


# In[ ]:


#no. of values for each quality 
sns.catplot(x='quality',data = wine_dataset, kind = 'count')


# In[ ]:


#volatile acidity vs quality coulmns 
plot = plt.figure(figsize=(8,6))
sns.barplot(x='quality',y='volatile acidity',data = wine_dataset)


# In[ ]:


#citric acid vs quality coulmns 
plot = plt.figure(figsize=(8,6))
sns.barplot(x='quality',y='citric acid',data = wine_dataset)


# # Corelation

# In[ ]:


correlation = wine_dataset.corr()


# In[ ]:


#construting heat map to understand correaltion 
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'BuPu')


# # Data Preprocessing 

# In[ ]:


#seperate data and labels
X = wine_dataset.drop('quality',axis=1)


# In[ ]:


print(X)


# # Label Binarization

# In[ ]:


Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>= 7 else 0)
print(Y)


# # Training and Testing split

# In[ ]:


X_train ,X_test ,Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[ ]:


print(Y.shape,Y_train.shape,Y_test.shape)


# # Random Test Classifier

# In[ ]:


model = RandomForestClassifier()


# In[ ]:


model.fit(X_train ,Y_train)


# # Model Evalution 

# Accuracy Score 
# 

# In[ ]:


#accuracy on data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[ ]:


print('Accuracy :',test_data_accuracy)


# # Buliding a predictive system 

# In[ ]:


input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

#changing input to numpy array

input_data_as_numpy_array =np.array(input_data)

#reshape the data as we're predicting the label for only one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)


# In[ ]:


if (prediction[0]==1):
  print("The Quality is GOOD")
else:
  print("The Quality is BAD")
  


# If we change the input in input_data we can predict. 
