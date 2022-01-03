# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
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
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score
from sklearn.ensemble import AdaBoostClassifier
import pickle
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Features
    Y: Target
    """
    # load data from database 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("InsertTableName", engine)
    x = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    y.related.replace(2,1,inplace=True)
    return x, y, category_names 

def tokenize(text):
    """
    Tokenizes and lemmatizes text.
    
    Parameters:
    text: Text to be tokenized
    
    Returns:
    clean_tokens: Returns cleaned tokens 
    """
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #iterate through each token
    clean_tokens=[]
    for tok in tokens:
        # lemmatize, normalise case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    '''
    Builds a machine learning pipeline to train. Uses Grid Search to find the best parameters.
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=10)))
        ])
    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'tfidf__use_idf': (True, False)
        }

    #cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    '''
    Calculates the precision, recall and f1-score for each of the category names
    '''
    y_pred = model.predict(X_test)
    evaluation = classification_report(y_test, y_pred, target_names=category_names)
    print(evaluation)
        
        
def save_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Builds the model, trains the model, evaluates the model, saves the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y,category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test,category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()