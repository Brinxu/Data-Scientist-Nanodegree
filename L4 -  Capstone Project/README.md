## Capstone Project: Predicting Patients No-Shows

### Project Overview

In this project, I seek to understand how likely is a patient with certain attributes, to show up to his hospital appointment. 

I investigate a data set of appointment records in public hospitals in Vitoria, Espirito Santo, Brazil. The data includes whether the patient showed up to the appointment, which is the main focus, as well as other attributes of the patient and the appointment. A link to the dataset can be downloaded [here](https://www.kaggle.com/joniarroba/noshowappointments)

I first do clean the data and explore exploratory analysis. I then train classification models using DecisionTree, a RandomForest, a linear SVM, an SVM with a radial basis kernel, AdaBoost. Finally, I resemble and blend models to predict the target variable. By comparing models' AUC and accuracy, I identify the best model in predicting patient no-shows.

### File Description

There is one exploratory notebook and html file of the notebook available here to showcase my work in predicting churn. Markdown cells were used throughout to explain the process taken.

The main findings of the code can be found at the Medium Blog post available here explaining the technical details of my project: https://medium.com/@brinxu1/predicting-patient-no-shows-790be6999b40

### Main Findings

Through this analysis, the Decision Tree method yields the highest AUC, while three models (Linear SVC, Random Forest, Blending) have the highest Accuracy. Considering both measurements, it looks like the Resembling model would stand out because of its performance relatively well in both measurements. The comparison of all models are below: 

| Classifier | AUC score | Accuracy |
| --- | --- | --- |
| RF | .5 | .79 |
| DT | .51 | .73 |
| Linear SVC | .5 | .80 |
| Rbf SVC | .5 | .80 |
| RF (With tuned parameters) | .56 | .79 |
| DT (With tuned parameters) | .60 | .73 |
| AdaBoost | .50 | .79 |
| Blending | .53 | .80 |

### Acknowledgment

This is my Capstone Project for the Udacity Data Science Nanodegree. Through my intense study in the program, I learned comprehensive knowledge and hands-on skills in data science. This has been a great experience. I have been benefiting from the Udacity instructor and mentor team’s support. I especially appreciate a technical mentor, Survesh’s help. 
