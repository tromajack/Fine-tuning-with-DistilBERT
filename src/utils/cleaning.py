import typing
import re
import string
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.utils.logging import app_logger


def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Input text string.
        
    Returns:
        Cleaned text string.
    """
    try:
        # Convert to lowercase
        text = text.lower()

        # Remove URLs/DOBs (where DOBs can have letters or digits)
        text = re.sub(r'http\S+|www\S+|https\S+|[a-zA-Z0-9]{2}/[a-zA-Z0-9]{2}/[a-zA-Z0-9]{4}', '', text, flags=re.MULTILINE)

        text = re.sub(r'\bxxxx+\b', '', text)

        # Tokenize
        tokens = word_tokenize(text)
        
      
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        stop_words.update(['xxxx', '--'])
        stop_words.update(string.punctuation)
        stop_words.update(['"', "'", '(', ')', '[', ']', '{', '}', '...', '``', "''"])
    
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        app_logger.error(f"Error in text cleaning: {e}")

def label_encoding(df):
    """
    Merge into common categories.
    
    Args:
        labels: List of string labels.
        
    Returns:
        List of str-encoded labels.
    """
    df['label'] = df['label'].replace({'Credit reporting, credit repair services, or other personal consumer reports': 'credit_reporting',
                       'Debt collection': 'debt_collection',
                       'Credit reporting or other personal consumer reports': 'credit_reporting',
                       'Credit card or prepaid card': 'credit_card',
                       'Mortgage': 'mortgages_and_loans',
                       'Payday loan, title loan, personal loan, or advance loan': 'mortgages_and_loans',
                       'Checking or savings account': 'retail_banking',
                       'Money transfer, virtual currency, or money service': 'retail_banking',
                       'Vehicle loan or lease': 'mortgages_and_loans',
                       'Payday loan, title loan, or personal loan': 'mortgages_and_loans',
                       'Student loan': 'mortgages_and_loans'})
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    return df
    