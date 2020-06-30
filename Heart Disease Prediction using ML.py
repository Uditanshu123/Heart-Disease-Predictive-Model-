#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings('ignore')


# In[97]:


df=pd.read_csv('heart.csv')


# In[98]:


df.head()


# In[99]:


df.dtypes


# In[100]:


df.isnull().sum()


# In[101]:


df.info()


# In[102]:


df.describe()


# In[103]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.1f')
plt.show()


# In[104]:


#Age Analysis


# In[105]:


df.age.value_counts()[:10]


# In[106]:


df['age'].value_counts()[:10].plot.bar()


# In[107]:


sns.barplot(x=df.age.value_counts()[:10].index,y=df.age.value_counts()[:10].values)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Analysis")
plt.show()


# In[108]:


df.target.value_counts()


# In[109]:


countnodisease=len(df[df.target==0])
countyesdisease=len(df[df.target==1])
print("people with no disease:{:.2f}%".format(((countnodisease)/len(df.target))*100))
print("people with disease:{:.2f}%".format(((countyesdisease)/len(df.target))*100))


# In[110]:


df.columns


# In[111]:


df['sex'].value_counts()


# In[112]:


noofmales=len(df[df.sex==1])
nooffemales=len(df[df.sex==0])


# In[113]:


print("No of males patients:{:.2f}%".format(((noofmales)/len(df.sex))*100))
print("No of males patients:{:.2f}%".format(((nooffemales)/len(df.sex))*100))


# In[114]:


#We use single AND in datframes


# In[115]:


young_ages=df[(df.age>=29)&(df.age<40)]
middle_ages=df[(df.age>=40)&(df.age<55)]
old_ages=df[df.age>=55]


# In[116]:


print('young ages',len(young_ages))
print('middle ages',len(middle_ages))
print('old ages',len(old_ages))


# In[117]:


colors=['blue','green','red']
explode=[1,1,1]
plt.figure(figsize=(8,8))
plt.pie([len(young_ages),len(middle_ages),len(old_ages)],labels=['young_ages','middle_ages','old_ages'])
plt.show()


# In[118]:


#chestpain analysis


# In[119]:


df.cp.value_counts()


# In[120]:


df.target.unique()


# In[121]:


sns.countplot(df.target)
plt.xlabel('target')
plt.ylabel('count')
plt.title('Target 1 4 0')
plt.show()


# In[122]:


df.corr()


# In[123]:


#Model building


# In[124]:


from sklearn.linear_model import LogisticRegression


# In[125]:


x_data=df.drop('target',axis=1)


# In[126]:


y=df['target']


# In[132]:


x_train,x_test,y_train,y_test=train_test_split(x_data,y,test_size=0.2,random_state=0)


# In[141]:


y_train.value_counts()


# In[142]:


y_test.value_counts()


# In[133]:


lreg=LogisticRegression()


# In[134]:


lreg.fit(x_train,y_train)


# In[138]:


print('Test_accuracy is:{:.2f}%'.format(lreg.score(x_test,y_test)*100))


# In[143]:


#Test_accuracy is:85.25% using logistic regression {quite-good:)}


# In[146]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)


# In[147]:


knn.fit(x_train,y_train)


# In[150]:


print('Test accuracy is:{:.2f}%'.format(knn.score(x_test,y_test)*100))


# In[151]:


#Test accuracy is:63.93% using knn model {quite-low :(}


# In[153]:


from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(x_train,y_train)


# In[156]:


print('Test accuracy is:{:.2f}%'.format(svm.score(x_test,y_test)*100))


# In[157]:


#Test accuracy is:59.02% using Support vector Machines {'quite-bad'}


# In[158]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)


# In[161]:


print('Test accuracy is:{:.2f}%'.format(clf.score(x_test,y_test)*100))


# In[162]:


#Test accuracy is:85.25% using naive_bayes algorithm{quite-good}


# In[166]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=1000,random_state=1)
rf.fit(x_train,y_train)


# In[168]:


print('Test accuracy is:{:.2f}%'.format(rf.score(x_test,y_test)*100))


# In[169]:


#Test accuracy is:85.25% using random forest {quite-good}


# In[ ]:




