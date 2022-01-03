# Disaster Response Pipeline Project

## Project Motivation
In this project, I analyze disaster data from Figure Eight to build models that classify disaster messages. A machine learning pipeline is created to categorize these events to make sure that messages would be properly classified. Finally, I include a web app where an emergency worker can input a new message and get classification results in several categories.

## File Description

Udacity Workspace Documents: 

![Screen Shot 2022-01-03 at 4 31 48 PM](https://user-images.githubusercontent.com/41206996/147985516-3655e96f-7d51-46d8-9948-6565f508a781.png)

ML pipeline notebook

ETL pipeline notebook

README.md

## Instruction
1. Run the following commands in the project's root directory to set up your database and model.

   To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

   To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/

## Acknowledgment 
This project benefits from the support from Udacity mentor and instructor team. I espcially appreciate Survesh's mentor help.
