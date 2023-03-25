#!/usr/bin/env python
# coding: utf-8

# # Import Scikit learn library

# In[1]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# # Understanding the Data

# In[2]:


print(iris_dataset.keys())


# In[3]:


print('Sample of Dataset \n{}'.format(iris_dataset['data'][:5]))


# In[4]:


print('Targets \n{}'.format(iris_dataset['target_names']))


# In[5]:


print(iris_dataset['DESCR'][:1100])


# In[6]:


print('Shape of data: {}'.format(iris_dataset['data'].shape))


# In[7]:


print('Shape of target: {}'.format(iris_dataset['target'].shape))


# In[8]:


print('Target \n{}'.format(iris_dataset['target']))
print('0 -> setosa\n1 -> versicolor\n2 -> virginica')


# ## Splitting Data

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[10]:


print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))


# In[11]:


print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))


# ## Visualizing Data

# In[24]:


import pandas as pd
import mglearn
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', 
                       hist_kwds={'bins': 20}, s=60, alpha=0.8, cmap=mglearn.cm3)


# ## Training Model

# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                    weights='uniform')


# ## Prediction

# In[32]:


import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])  ##two brackets because X is a matrix as scikit-learn
                                      ##always expects tow-dimensional arrays for data
print('X_new.shape: {}'.format(X_new.shape))


# In[36]:


prediction = knn.predict(X_new)
print('Prediction: {}'.format(prediction))
print('Predicted target name: {}'.format(iris_dataset['target_names'][0]))


# ## Model Evaluation

# In[40]:


y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))


# In[44]:


print("Test set score: {:0.2f}".format(np.mean(y_pred == y_test)))


# In[45]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

