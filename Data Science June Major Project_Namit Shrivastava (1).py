#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing the File

# In[2]:


df1 = pd.read_csv('adult.csv')
df1.head()


# # Renaming the columns

# In[3]:


df1.columns


# In[4]:


df1_cols = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']


# In[5]:


df1.columns = df1_cols


# In[6]:


df1.head()


# In[7]:


df1.tail()


# # Checking Null Values

# In[8]:


df1.isnull().sum()


# # Calculating stats of each column

# In[9]:


df1.describe()


# # Training and testing data Split And Transforming the Classification Columns using LabelEncoder

# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


lb = LabelEncoder() 
df1['Workclass'] = lb.fit_transform(df1['Workclass']) 
df1['Education'] = lb.fit_transform(df1['Education'])
df1['marital_status'] = lb.fit_transform(df1['marital_status'])
df1['occupation'] = lb.fit_transform(df1['occupation'])
df1['relationship'] = lb.fit_transform(df1['relationship'])
df1['race'] = lb.fit_transform(df1['race'])
df1['sex'] = lb.fit_transform(df1['sex'])
df1['native_country'] = lb.fit_transform(df1['native_country'])


# In[12]:


x = df1[['Age', 'Workclass', 'Fnlwgt', 'Education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']]
y = df1['income']
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Decision Tree Classifier

# In[15]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[16]:


def gen_metrics(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('confusion matrix\n',cm)
    print('Classification report\n',classification_report(ytest,ypred))
    print('Acc Score', accuracy_score(ytest,ypred))


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


m1 = DecisionTreeClassifier()
m1.fit(x_train,y_train)


# In[19]:


#Accuracy
print('Training Score',m1.score(x_train,y_train))
print('Testing Score',m1.score(x_test,y_test))


# In[20]:


ypred_m1 = m1.predict(x_test)
print(ypred_m1)


# In[21]:


gen_metrics(y_test,ypred_m1)


# # Validation

# In[22]:


pre0 = 5416/(5416+741)
pre1 = 1207/(1207+776)
rec0 = 5416/(5416+776)
rec1 = 1207/(1207+741)
print('Pre0',pre0)
print('Pre1',pre1)
print('Rec0',rec0)
print('Rec1',rec1)


# In[23]:


f1s0 = 2*pre0*rec0/(pre0+rec0)
f1s1 = 2*pre1*rec1/(pre1+rec1)
print('F1_Score0',f1s0)
print('F1_Score1',f1s1)


# In[24]:


acc = (5416+1207)/(5416+776+741+1207)
print('Accuracy',acc)


# # Percentage of Misclassification

# In[25]:


# Misclassification Rate = incorrect predictions /  total predictions
# Misclassification Rate = (false positive + false negative) / (total predictions)
MR = (741+776)/(5416+776+741+1207)
print('Percentage of Misclassification : ',MR*100, '%')


# # Random Forest Classifier

# In[26]:


from sklearn.ensemble import RandomForestClassifier


# In[27]:


m2 = RandomForestClassifier()
m2.fit(x_train,y_train)


# In[28]:


print('Training Score',m2.score(x_train,y_train))
print('Testing Score',m2.score(x_test,y_test))


# In[29]:


ypred_m2 = m2.predict(x_test)
print(ypred_m2)


# In[30]:


gen_metrics(y_test,ypred_m2)


# # Validation

# In[31]:


pre0 = 5803/(5803+746)
pre1 = 1202/(1202+389)
rec0 = 5803/(5803+389)
rec1 = 1202/(1202+746)
print('Pre0',pre0)
print('Pre1',pre1)
print('Rec0',rec0)
print('Rec1',rec1)


# In[32]:


f1s0 = 2*pre0*rec0/(pre0+rec0)
f1s1 = 2*pre1*rec1/(pre1+rec1)
print('F1_Score0',f1s0)
print('F1_Score1',f1s1)


# In[33]:


acc = (5803+1202)/(5803+389+746+1202)
print('Accuracy',acc)


# # Percentage of Misclassification

# In[34]:


# Misclassification Rate = incorrect predictions /  total predictions
# Misclassification Rate = (false positive + false negative) / (total predictions)
MR = (746+389)/(5803+389+746+1202)
print('Percentage of Misclassification : ',MR*100, '%')


# # KNN Classifier

# In[35]:


from sklearn.neighbors import KNeighborsClassifier


# In[36]:


m3 = KNeighborsClassifier()
m3.fit(x_train,y_train)


# In[37]:


#Accuracy
print('Training Score',m3.score(x_train,y_train))
print('Testing Score',m3.score(x_test,y_test))


# In[38]:


ypred_m3 = m3.predict(x_test)
print(ypred_m3)


# In[39]:


gen_metrics(y_test,ypred_m3)


# # Validation

# In[40]:


pre0 = 5704/(5704+1345)
pre1 = 603/(603+488)
rec0 = 5704/(5704+488)
rec1 = 603/(603+1345)
print('Pre0',pre0)
print('Pre1',pre1)
print('Rec0',rec0)
print('Rec1',rec1)


# In[41]:


f1s0 = 2*pre0*rec0/(pre0+rec0)
f1s1 = 2*pre1*rec1/(pre1+rec1)
print('F1_Score0',f1s0)
print('F1_Score1',f1s1)


# In[42]:


acc = (5704+603)/(5704+488+1345+603)
print('Accuracy',acc)


# # Percentage of Misclassification

# In[43]:


# Misclassification Rate = incorrect predictions /  total predictions
# Misclassification Rate = (false positive + false negative) / (total predictions)
MR = (1345+488)/(5704+488+1345+603)
print('Percentage of Misclassification : ',MR*100, '%')


# # Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression


# In[45]:


m4 = LogisticRegression(max_iter=10000)
m4.fit(x_train,y_train)


# In[46]:


#Accuracy
print('Training Score',m4.score(x_train,y_train))
print('Testing Score',m4.score(x_test,y_test))


# In[47]:


ypred_m4 = m4.predict(x_test)
print(ypred_m4)


# In[48]:


gen_metrics(y_test,ypred_m4)


# In[49]:


m = m4.coef_
c = m4.intercept_
print('Coefficient',m)
print('Intercept or constant',c)


# In[50]:


print('Testing Score',m4.score(x_test,y_test))
print('Accuracy Score',accuracy_score(y_test,ypred_m4))


# In[51]:


def sigmoid(X,m,c):
    logit = 1/(1 + np.exp(-(m*X+c)))
    print(logit)


# In[52]:


sigmoid (0.077316,m,c) #example using any value taken as X


# # Validation

# In[53]:


pre0 = 5854/(5854+1374)
pre1 = 574/(574+338)
rec0 = 5854/(5854+338)
rec1 = 574/(574+1374)
print('Pre0',pre0)
print('Pre1',pre1)
print('Rec0',rec0)
print('Rec1',rec1)


# In[54]:


f1s0 = 2*pre0*rec0/(pre0+rec0)
f1s1 = 2*pre1*rec1/(pre1+rec1)
print('F1_Score0',f1s0)
print('F1_Score1',f1s1)


# In[55]:


acc = (5854+574)/(5854+338+1374+574)
print('Accuracy',acc)


# # Percentage of Misclassification

# In[56]:


# Misclassification Rate = incorrect predictions /  total predictions
# Misclassification Rate = (false positive + false negative) / (total predictions)
MR = (1374+338)/(5854+338+1374+574)
print('Percentage of Misclassification : ',MR*100, '%')


# # SVM Classifier

# In[57]:


from sklearn.svm import LinearSVC


# In[58]:


m5 = LinearSVC(random_state=0, tol=1e-5, dual=False)
m5.fit(x_train, y_train.ravel())


# In[59]:


#Accuracy
print('Training Score',m5.score(x_train,y_train))
print('Testing Score',m5.score(x_test,y_test))


# In[60]:


ypred_m5 = m5.predict(x_test)
print(ypred_m5)


# In[61]:


gen_metrics(y_test,ypred_m5)


# # Validation

# In[62]:


pre0 = 5933/(5933+1330)
pre1 = 618/(618+259)
rec0 = 5933/(5933+259)
rec1 = 618/(618+1330)
print('Pre0',pre0)
print('Pre1',pre1)
print('Rec0',rec0)
print('Rec1',rec1)


# In[63]:


f1s0 = 2*pre0*rec0/(pre0+rec0)
f1s1 = 2*pre1*rec1/(pre1+rec1)
print('F1_Score0',f1s0)
print('F1_Score1',f1s1)


# In[64]:


acc = (5933+618)/(5933+259+1330+618)
print('Accuracy',acc)


# # Percentage of Misclassification

# In[65]:


# Misclassification Rate = incorrect predictions /  total predictions
# Misclassification Rate = (false positive + false negative) / (total predictions)
MR = (1330+259)/(5933+259+1330+618)
print('Percentage of Misclassification : ',MR*100, '%')


# # CONCLUSION: The Accuracy Score of each model is as follows:
# 
# 
# a. Decision Tree Classifier : Acc Score Accuracy 0.8136363636363636<br>
# b. Random Forest Classifier : Accuracy 0.8605651105651105<br>
# c. KNN Classifier : Accuracy 0.7748157248157248<br>
# d. Logistic Regression : Accuracy 0.7896805896805896<br>
# e. SVM Classifier : Accuracy 0.8047911547911548<br>
# 
# Hence the model with the highest accuracy is Random Forest Classifier with an accuracy score of 0.8605651105651105
