import pandas as pd
import numpy
from sklearn import pipeline

FILEPATH = "../data/raw/insurance_claims.csv"

def load_data(FILEPATH):
    return pd.read_csv(FILEPATH)

def clean_data(data:pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    # Handling of duplicate data
    if cleaned_data.duplicated().any():
        cleaned_data = cleaned_data.drop_duplicates()
    # Handling of outliers

    # Handling of erroneus data

    # handling missing data

    
    return cleaned_data
    

def preprocess(data):
    pass
