# Disaster Response Pipeline Project

## Project Motivation
In this project, I analyze disaster data from Figure Eight to build models that classify disaster messages. A machine learning pipeline is created to categorize these events to make sure that messages would be properly classified. Finally, I include a web app where an emergency worker can input a new message and get classification results in several categories.

## File Description

Udacity Workspace Documents: 
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # data cleaning pipeline
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py # machine learning pipeline
|- classifier.pkl # saved model

Notebooks run in a local machne:
ML pipeline notebook
ETL pipeline notebook

README.md

## Instruction
1. Run the following commands in the project's root directory to set up your database and model.
To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app. python run.py
3. 
4. Go to http://0.0.0.0:3001/

## Acknowledgment 
This project greatly benefits from the Udacity mentor and instructor's help. I espcially appreciate my technical mentor, Survesh's patient and helpful support. 
