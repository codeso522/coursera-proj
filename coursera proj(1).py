#!/usr/bin/env python
# coding: utf-8

# # logistic regression

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[2]:


ds = pd.read_csv(r"https://raw.githubusercontent.com/codeso522/coursera-proj/master/loan_data.csv")


# In[3]:


ds.head()


# In[4]:


ds.drop('Loan_ID',axis=1,inplace=True)


# In[5]:


ds.head()


# In[6]:


sns.pairplot(ds)


# In[7]:


ds.isnull().sum()


# In[8]:


ds.shape


# In[9]:


ds.drop('Credit_History',axis=1,inplace=True)


# In[10]:


ds.drop('Loan_Amount_Term',axis=1,inplace=True)


# In[11]:


ds.head(5)


# In[12]:


ds.drop('Self_Employed',axis=1,inplace=True)


# In[13]:


ds.isnull().sum()


# In[14]:


ds.drop('Gender',axis=1,inplace=True)


# In[15]:


ds.drop('Dependents',axis=1,inplace=True)


# In[16]:


ds.isnull().sum()


# In[17]:


ds.dropna(inplace=True)


# In[18]:


ds.isnull().sum()


# In[19]:


ds.head()


# In[20]:


married=pd.get_dummies(ds['Married'],drop_first=True)
married.head(5)


# In[21]:


edu=pd.get_dummies(ds['Education'],drop_first=True)


# In[22]:


edu.head(5)


# In[23]:


area=pd.get_dummies(ds['Property_Area'],drop_first=True)
area.head(5)


# In[24]:


loan=pd.get_dummies(ds['Loan_Status'],drop_first=True)


# In[25]:


ds=pd.concat([ds,married,edu,area,loan],axis=1)


# In[26]:


ds.head(5)


# In[27]:


ds.drop(['Married','Education','Property_Area','Loan_Status'],axis=1,inplace=True)


# In[28]:


ds.head(5)


# In[29]:


X=ds.iloc[:,:-1].values


# In[30]:


y=ds.iloc[:,-1].values


# In[31]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[33]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[34]:


y_pred=logmodel.predict(X_test)


# In[35]:


from sklearn.metrics import classification_report
classification_report(y_test,y_pred)


# In[36]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[37]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# # Knn algorithm

# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[39]:


ds = pd.read_csv(r"https://raw.githubusercontent.com/codeso522/coursera-proj/master/loan_data.csv")
ds.head(5)


# In[40]:


ds.drop('Loan_ID',axis=1,inplace=True)


# In[41]:


ds.isnull().sum()


# In[42]:


ds.drop('Credit_History',axis=1,inplace=True)


# In[43]:


ds.dropna(inplace=True)


# In[44]:


ds.head(5)


# In[45]:


ds.isnull().sum()


# In[46]:


married=pd.get_dummies(ds['Married'],drop_first=True)
area=pd.get_dummies(ds['Property_Area'],drop_first=True)
edu=pd.get_dummies(ds['Education'],drop_first=True)
loan=pd.get_dummies(ds['Loan_Status'],drop_first=True)
gender=pd.get_dummies(ds['Gender'],drop_first=True)
emp=pd.get_dummies(ds['Self_Employed'],drop_first=True)
Dependents=pd.get_dummies(ds['Dependents'],drop_first=True)


# In[47]:


ds=pd.concat([ds,married,area,edu,loan,gender,emp,Dependents],axis=1)
ds.head(5)


# In[48]:


ds.drop(['Gender','Married','Education','Self_Employed','Property_Area'],axis=1,inplace=True)


# In[49]:


ds.head(5)


# In[50]:


ds.drop(['Dependents'],axis=1,inplace=True)


# In[51]:


ds.head()


# In[52]:


ds.drop(['Loan_Status'],axis=1,inplace=True)


# In[53]:


ds.head(5)


# In[54]:


X=ds.iloc[:,:-8].values
y=ds.iloc[:,8].values


# In[55]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[56]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[57]:


# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p= 2)
classifier.fit(X_train,y_train)


# In[58]:


y_pred = classifier.predict(X_test)


# In[59]:


print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))


# In[60]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[61]:


cm


# # Support vector machine

# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[63]:


ds = pd.read_csv(r"https://raw.githubusercontent.com/codeso522/coursera-proj/master/loan_data.csv")
ds.head(5)


# In[64]:


ds.drop('Loan_ID',axis=1,inplace=True)
ds.drop('Credit_History',axis=1,inplace=True)
ds.dropna(inplace=True)


# In[65]:


married=pd.get_dummies(ds['Married'],drop_first=True)
area=pd.get_dummies(ds['Property_Area'],drop_first=True)
edu=pd.get_dummies(ds['Education'],drop_first=True)
loan=pd.get_dummies(ds['Loan_Status'],drop_first=True)
gender=pd.get_dummies(ds['Gender'],drop_first=True)
emp=pd.get_dummies(ds['Self_Employed'],drop_first=True)
Dependents=pd.get_dummies(ds['Dependents'],drop_first=True)


# In[66]:


ds=pd.concat([ds,married,area,edu,loan,gender,emp,Dependents],axis=1)
ds.head(5)


# In[67]:


ds.drop(['Gender','Married','Education','Self_Employed','Property_Area'],axis=1,inplace=True)


# In[68]:


ds.drop(['Dependents'],axis=1,inplace=True)


# In[69]:


ds.drop(['Loan_Status'],axis=1,inplace=True)


# In[70]:


X=ds.iloc[:,:-8].values
y=ds.iloc[:,8].values


# In[71]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[72]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[73]:


# Fitting svm to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,y_train)


# In[74]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[75]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[76]:


cm


# In[77]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)


# # Decision tree

# In[78]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[79]:


ds = pd.read_csv(r"https://raw.githubusercontent.com/codeso522/coursera-proj/master/loan_data.csv")
ds.head(5)


# In[80]:


ds.drop('Loan_ID',axis=1,inplace=True)
ds.drop('Credit_History',axis=1,inplace=True)
ds.dropna(inplace=True)


# In[81]:


married=pd.get_dummies(ds['Married'],drop_first=True)
area=pd.get_dummies(ds['Property_Area'],drop_first=True)
edu=pd.get_dummies(ds['Education'],drop_first=True)
loan=pd.get_dummies(ds['Loan_Status'],drop_first=True)
gender=pd.get_dummies(ds['Gender'],drop_first=True)
emp=pd.get_dummies(ds['Self_Employed'],drop_first=True)
Dependents=pd.get_dummies(ds['Dependents'],drop_first=True)


# In[82]:


ds=pd.concat([ds,married,area,edu,loan,gender,emp,Dependents],axis=1)
ds.head(5)


# In[83]:


ds.drop(['Gender','Married','Education','Self_Employed','Property_Area'],axis=1,inplace=True)


# In[84]:


ds.drop(['Dependents'],axis=1,inplace=True)


# In[85]:


ds.drop(['Loan_Status'],axis=1,inplace=True)


# In[86]:


X=ds.iloc[:,:-8].values
y=ds.iloc[:,8].values


# In[87]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[88]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[89]:


#fitting the decision tree regression model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor =  DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


# In[90]:


#predicting a new result
y_pred = regressor.predict(X_test)


# In[91]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[92]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

