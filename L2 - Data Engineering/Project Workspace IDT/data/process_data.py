import sys
import pandas as pd
from sqlalchemy import *

def load_data(messages_filepath, categories_filepath):
    
    '''
    Merges the data from messages and categories csv files on id and returns a dataframe
    '''
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets on common id and assign to df
    df = pd.merge(messages, categories, on='id', how='outer')
    return df

def clean_data(df):
    
    '''
    Cleans the dataframe.
    '''
    # create a dataframe including individual category columns
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0]
    # apply a lambda function that takes everything
    category_colnames = list(map(lambda x: x.split("-")[0], categories.iloc[0].values.tolist()))
    # rename the columns 
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)
    del df['categories']
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='first')
    return df
   

def save_data(df, database_filename):
    '''Saves the cleaned dataframe to a table messages in the database given'''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('InsertTableName', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()