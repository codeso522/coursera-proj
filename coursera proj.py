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


# In[8]:


ds.isnull().sum()


# In[10]:


ds.shape


# In[11]:


ds.drop('Credit_History',axis=1,inplace=True)


# In[12]:


ds.drop('Loan_Amount_Term',axis=1,inplace=True)


# In[13]:


ds.head(5)


# In[14]:


ds.drop('Self_Employed',axis=1,inplace=True)


# In[15]:


ds.isnull().sum()


# In[16]:


ds.drop('Gender',axis=1,inplace=True)


# In[17]:


ds.drop('Dependents',axis=1,inplace=True)


# In[18]:


ds.isnull().sum()


# In[19]:


ds.dropna(inplace=True)


# In[20]:


ds.isnull().sum()


# In[21]:


ds.head()


# In[23]:


married=pd.get_dummies(ds['Married'],drop_first=True)
married.head(5)


# In[24]:


edu=pd.get_dummies(ds['Education'],drop_first=True)


# In[25]:


edu.head(5)


# In[26]:


area=pd.get_dummies(ds['Property_Area'],drop_first=True)
area.head(5)


# In[27]:


loan=pd.get_dummies(ds['Loan_Status'],drop_first=True)


# In[28]:


ds=pd.concat([ds,married,edu,area,loan],axis=1)


# In[29]:


ds.head(5)


# In[30]:


ds.drop(['Married','Education','Property_Area','Loan_Status'],axis=1,inplace=True)


# In[31]:


ds.head(5)


# In[32]:


X=ds.iloc[:,:-1].values


# In[36]:


y=ds.iloc[:,-1].values


# In[37]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[38]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[39]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[40]:


y_pred=logmodel.predict(X_test)


# In[41]:


from sklearn.metrics import classification_report
classification_report(y_test,y_pred)


# In[42]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[43]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# # Knn algorithm

# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[47]:


ds = pd.read_csv(r"https://raw.githubusercontent.com/codeso522/coursera-proj/master/loan_data.csv")
ds.head(5)


# In[48]:


ds.drop('Loan_ID',axis=1,inplace=True)


# In[49]:


ds.isnull().sum()


# In[50]:


ds.drop('Credit_History',axis=1,inplace=True)


# In[51]:


ds.dropna(inplace=True)


# In[52]:


ds.head(5)


# In[53]:


ds.isnull().sum()


# In[54]:


married=pd.get_dummies(ds['Married'],drop_first=True)
area=pd.get_dummies(ds['Property_Area'],drop_first=True)
edu=pd.get_dummies(ds['Education'],drop_first=True)
loan=pd.get_dummies(ds['Loan_Status'],drop_first=True)
gender=pd.get_dummies(ds['Gender'],drop_first=True)
emp=pd.get_dummies(ds['Self_Employed'],drop_first=True)
Dependents=pd.get_dummies(ds['Dependents'],drop_first=True)


# In[55]:


ds=pd.concat([ds,married,area,edu,loan,gender,emp,Dependents],axis=1)
ds.head(5)


# In[56]:


ds.drop(['Gender','Married','Education','Self_Employed','Property_Area'],axis=1,inplace=True)


# In[57]:


ds.head(5)


# In[59]:


ds.drop(['Dependents'],axis=1,inplace=True)


# In[60]:


ds.head()


# In[61]:


ds.drop(['Loan_Status'],axis=1,inplace=True)


# In[62]:


ds.head(5)


# In[65]:


X=ds.iloc[:,:-8].values
y=ds.iloc[:,8].values


# In[66]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[68]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[69]:


# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p= 2)
classifier.fit(X_train,y_train)


# In[70]:


y_pred = classifier.predict(X_test)


# In[71]:


print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))


# In[72]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[73]:


cm


# # Support vector machine

# In[76]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[77]:


ds = pd.read_csv(r"https://raw.githubusercontent.com/codeso522/coursera-proj/master/loan_data.csv")
ds.head(5)


# In[78]:


ds.drop('Loan_ID',axis=1,inplace=True)
ds.drop('Credit_History',axis=1,inplace=True)
ds.dropna(inplace=True)


# In[79]:


married=pd.get_dummies(ds['Married'],drop_first=True)
area=pd.get_dummies(ds['Property_Area'],drop_first=True)
edu=pd.get_dummies(ds['Education'],drop_first=True)
loan=pd.get_dummies(ds['Loan_Status'],drop_first=True)
gender=pd.get_dummies(ds['Gender'],drop_first=True)
emp=pd.get_dummies(ds['Self_Employed'],drop_first=True)
Dependents=pd.get_dummies(ds['Dependents'],drop_first=True)


# In[80]:


ds=pd.concat([ds,married,area,edu,loan,gender,emp,Dependents],axis=1)
ds.head(5)


# In[81]:


ds.drop(['Gender','Married','Education','Self_Employed','Property_Area'],axis=1,inplace=True)


# In[82]:


ds.drop(['Dependents'],axis=1,inplace=True)


# In[83]:


ds.drop(['Loan_Status'],axis=1,inplace=True)


# In[84]:


X=ds.iloc[:,:-8].values
y=ds.iloc[:,8].values


# In[85]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[86]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[87]:


# Fitting svm to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,y_train)


# In[88]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[89]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[90]:


cm


# In[91]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)


# # Decision tree

# In[92]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[93]:


ds = pd.read_csv(r"https://raw.githubusercontent.com/codeso522/coursera-proj/master/loan_data.csv")
ds.head(5)


# In[94]:


ds.drop('Loan_ID',axis=1,inplace=True)
ds.drop('Credit_History',axis=1,inplace=True)
ds.dropna(inplace=True)


# In[95]:


married=pd.get_dummies(ds['Married'],drop_first=True)
area=pd.get_dummies(ds['Property_Area'],drop_first=True)
edu=pd.get_dummies(ds['Education'],drop_first=True)
loan=pd.get_dummies(ds['Loan_Status'],drop_first=True)
gender=pd.get_dummies(ds['Gender'],drop_first=True)
emp=pd.get_dummies(ds['Self_Employed'],drop_first=True)
Dependents=pd.get_dummies(ds['Dependents'],drop_first=True)


# In[96]:


ds=pd.concat([ds,married,area,edu,loan,gender,emp,Dependents],axis=1)
ds.head(5)


# In[97]:


ds.drop(['Gender','Married','Education','Self_Employed','Property_Area'],axis=1,inplace=True)


# In[98]:


ds.drop(['Dependents'],axis=1,inplace=True)


# In[99]:


ds.drop(['Loan_Status'],axis=1,inplace=True)


# In[100]:


X=ds.iloc[:,:-8].values
y=ds.iloc[:,8].values


# In[101]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[102]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[104]:


#fitting the decision tree regression model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor =  DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


# In[106]:


#predicting a new result
y_pred = regressor.predict(X_test)


# In[107]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[108]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)


# In[ ]:




