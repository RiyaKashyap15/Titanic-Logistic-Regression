#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score,cohen_kappa_score
#from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('datasets_122398_315766_full.csv')


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.drop(df[['PassengerId', 'WikiId', 'Name_wiki',
       'Age_wiki', 'Hometown', 'Boarded', 'Destination', 'Lifeboat', 'Body',
       'Class']],axis=1,inplace=True)


# In[6]:


df.tail()


# In[7]:


# univariate analysis
#rate of survived is less than the rate of non survivors
sns.countplot(x='Survived',data = df,color='red')


# In[8]:


#more passengers were in class 3
sns.countplot(x='Pclass',data = df,color='green')


# In[9]:


sns.countplot(x='Sex',data = df,color='red')


# In[10]:


#more passengers were of age between 20-40
df['Age'].plot.hist(bins=20)


# In[11]:


sns.countplot(x='SibSp',data = df,color='red')


# In[12]:


sns.countplot(x='Parch',data = df,color='red')


# In[13]:


sns.boxplot(x='Fare',data=df,color='purple',linewidth=2)


# In[14]:


sns.countplot(x ='Sex',hue='Survived',data=df,color = 'red')


# In[15]:


sns.countplot(x ='Pclass',hue='Survived',data=df,color = 'purple')


# In[16]:


sns.countplot(x ='Parch',hue='Survived',data=df,color = 'green')


# In[17]:


sns.boxplot(x ='Survived',y='Age',data=df,color = 'orange')


# In[18]:


sns.countplot(x ='SibSp',hue='Survived',data=df,color = 'blue')


# In[19]:


sns.boxplot(x ='Survived',y='Fare',data=df,color = 'pink')


# df['Fare']=list(df['Fare'])
# new_fare=[]
# for i in df['Fare']:
#     if i<300:
#         new_fare.append(i)
# print(new_fare)

# new_fare=pd.DataFrame(new_fare)
# df['newfare']= new_fare

# In[20]:


df.dtypes


# In[21]:


df.isnull().sum()


# In[22]:


df = df[df['Survived'].notna()]


# In[23]:


df.corr()


# In[24]:


df.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)


# In[25]:


df.columns


# In[26]:


for col in ['Survived', 'Pclass', 'Sex','Embarked']:
    df[col]=df[col].astype('category')

df.dtypes


# In[27]:


cat_attr=list(df.select_dtypes('category').columns)
num_attr=list(df.columns.difference(cat_attr))
cat_attr.pop(0)


# In[28]:


x=df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']]
y=df['Survived']


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=2)


# In[30]:


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent',fill_value="missing_value")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_attr),
        ('cat', categorical_transformer, cat_attr)])


# In[31]:


clf_lr=Pipeline(steps=[('preprocessor',preprocessor),
                       ('model',LogisticRegression())])
clf_lr.fit(x_train,y_train)


# In[32]:


predict = clf_lr.predict(x_test)


# In[33]:


train_pred = clf_lr.predict(x_train)
test_pred = clf_lr.predict(x_test)
confusion_matrix_test = confusion_matrix(y_test, test_pred)
confusion_matrix_train = confusion_matrix(y_train, train_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# In[52]:


from sklearn.metrics import classification_report
print(classification_report(predictions,y_test))
print('recall_score :',recall_score(predict,y_test))
print('precision_score :',precision_score(predict,y_test))
print('accuracy_score :',accuracy_score(y_test,predict))


# In[36]:


import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[37]:


fpr, tpr, threshold = metrics.roc_curve(y_test, predict)
roc_auc = metrics.auc(fpr, tpr)
auc = roc_auc_score(y_test, predict)
print('AUC: %.2f' % auc)


# In[38]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# import statsmodels.api as sm
# 
# stats=Pipeline(steps=[('preprocessor',preprocessor),
#                       ('reg_mod',sm.OLS())])
# result = stats.fit(y_train,x_train)
# print(result.summary())

# In[40]:


np.asarray(x_train)


# In[41]:


np.asarray(y_train)


# In[42]:


clf_nb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GaussianNB())])
 

# Train the model using the training sets
clf_nb.fit(x_train,y_train)
train_predictions=clf_nb.predict(x_train)
predictions = clf_nb.predict(x_test)
print("train accuracy is:",accuracy_score(y_train,train_predictions))
print("test accuracy is:",accuracy_score(y_test, predictions))
cm=confusion_matrix(train_predictions,y_train)
cm = cm / cm.sum(axis=1)[:, np.newaxis]
cm.diagonal()


# In[53]:


from sklearn.metrics import classification_report
print(classification_report(predictions,y_test))
print('recall_score :',recall_score(predictions,y_test))
print('precision_score :',precision_score(predictions,y_test))
print('accuracy_score :',accuracy_score(y_test,predictions))


# In[ ]:




