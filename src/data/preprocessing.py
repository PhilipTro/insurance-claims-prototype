from pathlib import Path
import pandas as pd
import numpy
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data(file_path:Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(data:pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    # Handling of duplicate data
    if cleaned_data.duplicated().any():
        cleaned_data = cleaned_data.drop_duplicates()
    # Handling of outliers

    # Handling of erroneus data
    
    # handling missing data
    numeric_columns = cleaned_data.select_dtypes(include='number').columns
    cat_columns = cleaned_data.select_dtypes(include='object').columns
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cleaned_data[numeric_columns] = num_imputer.fit_transform(data[numeric_columns])
    cleaned_data[cat_columns] = cat_imputer.fit_transform(data[cat_columns])

    return cleaned_data
    

def split_data(data:pd.DataFrame, target:str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # Funcion for splitting into train_test_val
    X, y = data[[col for col in data.columns if col not in target]], data[target]

    #First split train vs test+val

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # Second split test vs val

    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size = 0.5)
    

    return X_train, X_test, X_val, y_train, y_test, y_val

