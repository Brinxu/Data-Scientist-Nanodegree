# Udacity-Data-Scientist-Nanodegree

This repository contains my projects for Udacity's Data Scientist Nanodegree. In this project, I took classes including **software engineering**, **data engineering**, **experimental design & Recommendation**. I gained comprehensive knowledge and hands-on experience in data science. Below is a list of projects I completed in the program. You can find each project under one folder. 

## [Project 1: Predicting loan Status](https://github.com/Brinxu/Data-Scientist-Nanodegree/tree/main/L1%20-%20Introduction%20to%20Data%20Science)

For this project, I was interested in conducting exploratory data analysis using a loan dataset found on Kaggle. I seek to understand what factors predict loan outcomes. My main findings are: 

For those with a home, I found that borrows' APR is the lowest for full-time employed.
For those without a home, it is one of the highest for those who do not have a home.
Putting together, it looks like those who are full-time employed and have a home enjoy the highest loan amount as well we the lowest borrower APR.

## [Project 2: Disaster Response Pipeline](https://github.com/Brinxu/Data-Scientist-Nanodegree/tree/main/L2%20-%20Data%20Engineering)

I applied my data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. I created a machine learning pipeline to categorize real messages that were sent during disaster events so that the messages could be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## [Project 3: Recommendations with IBM](https://github.com/Brinxu/Data-Scientist-Nanodegree/tree/main/L3%20-%20Experimental%20Design%20%26%20Recommendations)

I analyzed the interactions that users have with articles on the IBM Watson Studi platform and made recommendations to them about new articles I thought they'd like. I performed EDA, Rank Based Recommendations, User-user Based Collaborative Filtering, and Matrix factorization.

## [Project 4: Predicting Patients No-Shows](https://github.com/Brinxu/Data-Scientist-Nanodegree/tree/main/L4%20-%20%20Capstone%20Project)

In this project, I seek to understand how likely is a patient with certain attributes, to show up to his hospital appointment with medical record data at Kaggle. 

I first do clean the data and explore exploratory analysis. I then train classification models using DecisionTree, a RandomForest, a linear SVM, an SVM with a radial basis kernel, AdaBoost. Finally, I resemble and blend models to predict the target variable. By comparing models' AUC and accuracy, I identify the best model in predicting patient no-shows.

Through this analysis, the Decision Tree method yields the highest AUC, while three models (Linear SVC, Random Forest, Blending) have the highest Accuracy. Considering both measurements, it looks like the Resembling model would stand out because of its performance relatively well in both measurements.
