#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle


# In[2]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql("InsertTableName", engine)
df.head(1)


# In[3]:


x = df['message']
y = df.iloc[:, 4:]
y.head(1)


# ### 2. Write a tokenization function to process your text data

# In[4]:


def tokenize(text):
    
    """
    Function to tokenize text.
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[5]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(x, y)


# In[7]:


pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[8]:


y_pred = pipeline.predict(X_test)


# In[9]:


def test_model(y_test, y_pred):
    
    """
    Function to iterate through columns and call sklearn classification report on each.
    """
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, index]))


# In[10]:


test_model(y_test, y_pred)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[11]:


# use grid search to find better parameters

parameters = {
            'clf__estimator__n_estimators' : [50, 100]
             }
cv = GridSearchCV(pipeline, param_grid=parameters)


# In[12]:


cv.fit(X_train, y_train)


# In[13]:


y_pred = cv.predict(X_test)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[14]:


print("Accuracy")
print((y_pred == y_test).mean())

for i in range(35):
    print("Precision, Recall, F1 Score for {}".format(y_test.columns[i]))
    print(classification_report(y_test.iloc[:,i], y_pred[:,i]))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[15]:


# I try AdaBoostClassifier 

from sklearn.ensemble import AdaBoostClassifier
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])


# In[16]:


pipeline.fit(X_train, y_train)


# In[17]:


y_pred = pipeline.predict(X_test)


# In[18]:


for i in range(35):
    print("Precision, Recall, F1 Score for {}".format(y_test.columns[i]))
    print(classification_report(y_test.iloc[:,i], y_pred[:,i]))


# ### 9. Export your model as a pickle file

# In[19]:


filename = 'mlpipeline.pkl'
pickle.dump(pipeline, open(filename, 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




