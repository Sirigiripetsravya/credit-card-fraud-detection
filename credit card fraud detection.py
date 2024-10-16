#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec 


# In[4]:


data = pd.read_csv(r'C:\Users\USER\Downloads\creditcard.csv\creditcard.csv')
print(data)


# In[5]:


data.head()


# In[7]:


data.columns


# In[8]:


data.shape


# In[10]:


data.drop_duplicates(inplace= True)
data.shape


# In[11]:


data.isnull().sum()


# In[12]:


print(data.describe()) 


# In[13]:


fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction) 


# In[14]:


print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 


# In[15]:


print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# In[16]:


data[data['Class']==1]


# In[18]:


print('Amount details of the fraudulent transaction') 
fraud.Amount.describe() 


# In[19]:


print('details of valid transaction')
valid.Amount.describe() 


# In[20]:


corrmat = data.corr()
fig = plt.figure(figsize = (15,15 ))
sns.heatmap(data.corr(),cmap = 'RdYlGn',annot= False,center =0)
plt.show() 


# In[21]:


X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape) 


# In[22]:


# getting just the values for the sake of processing
# (its a numpy array with no columns)
xData = X.values
yData = Y.values 


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


xTrain, xTest, yTrain, yTest = train_test_split(
 xData, yData, test_size = 0.3, random_state = 42)
xTrain.shape


# In[26]:


xTest.shape


# In[27]:


from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
# predictions
yPred = rfc.predict(xTest) 


# In[28]:


# scoring in anything
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix 


# In[29]:


n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest classifier") 


# In[30]:


acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc)) 


# In[31]:


prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec)) 


# In[32]:


rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec)) 


# In[33]:


f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1)) 


# In[34]:


MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is{}".format(MCC)) 


# In[37]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS,
 yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()







