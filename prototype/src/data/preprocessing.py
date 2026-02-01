from itertools import dropwhile
from pathlib import Path
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

def to_int_pre(x):
    return x.astype(int)

def load_data(file_path:Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(data:pd.DataFrame) -> pd.DataFrame:
    """Performs rudimentary data cleaning for prototyping

    Args:
        data (pd.DataFrame): raw prototype dataset

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    cleaned_data = data.copy()
    # Handling of duplicate data
    if cleaned_data.duplicated().any():
        cleaned_data = cleaned_data.drop_duplicates()
    # Handling of outliers

    # Handling of erroneus data
    
    # handling missing data
    numeric_columns = cleaned_data.select_dtypes(include='number').columns
    cat_columns = cleaned_data.select_dtypes(include='object').columns
    bool_columns = []
    for col in data.columns:
        unique_vals = set(data[col].dropna().unique())
        if unique_vals.issubset({True, False}):
            data[col] = data[col].astype('boolean')  # nullable bool
            bool_columns.append(col)

    bool_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cleaned_data[numeric_columns] = num_imputer.fit_transform(cleaned_data[numeric_columns])
    cleaned_data[cat_columns] = cat_imputer.fit_transform(cleaned_data[cat_columns])
    if len(bool_columns) >= 1:
        cleaned_data[bool_columns] = bool_imputer.fit_transform(cleaned_data[bool_columns])
    date_columns = cleaned_data.select_dtypes(include='datetime').columns
    cleaned_data[bool_columns] = cleaned_data[bool_columns].astype('int64')
    for col in date_columns:
        cleaned_data[col + "_year"] = cleaned_data[col].dt.year # type: ignore
        cleaned_data[col + "_month"] = cleaned_data[col].dt.month # type: ignore
        cleaned_data[col + "_day"] = cleaned_data[col].dt.day # type: ignore
        cleaned_data[col + "_weekday"] = cleaned_data[col].dt.dayofweek # type: ignore

    return cleaned_data
    

def split_data(data:pd.DataFrame, target:str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits the data according to a train_val_test split. Where the ratios are 70-15-15 percent. 

    Args:
        data (pd.DataFrame): Raw dataset
        target (str): target feature (for either regression task or classification task)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: The split data set is returned as a tuple of X (dataset sans 
        target variable) and y (Target variable). A Xy pair is returned for each instance split. 
    """
    # Funcion for splitting into train_test_val
    X, y = data[[col for col in data.columns if col not in target]], data[target]

    #First split train vs test+val

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # Second split test vs val

    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size = 0.5)
    

    return X_train, X_test, X_val, y_train, y_test, y_val

def split_data_for_classification(data, target):
    t = data[target].quantile(0.8)
    data['severe'] = (data[target] >= t).astype(int)
    X, y = data[[col for col in data.columns if col not in (target, 'severe')]], data['severe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    print(y.value_counts(normalize=True))
    return X_train, X_test, y_train, y_test
def split_data_for_classification_fraud(data, target):
    X, y = data[[col for col in data.columns if col not in (target)]], data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    print(y.value_counts(normalize=True))
    return X_train, X_test, y_train, y_test


from imblearn.over_sampling import SMOTE

def oversampling(X, y):
    """Oversamples the minority class in the training data, uses automatic strategy. 

    Args:
        X (_type_): X Matrix
        y (_type_): y vector

    Returns:
        _type_: X Matrix (resamples), y vector (resampled)
    """
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X,y) #type: ignore
    return X_res, y_res

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce

def prepreprocessing_pipeline(X, numerical_cols, ordinal_cols, ordinal_categories, nominal_cols, bool_cols):
    """Creates a flexible preprocessing pipeline meant to handle numerical, ordinal, nominal and boolean values. 
       The pipeline is created using sklearn.pipeline.Pipeline(), where a pipeline is created for each type of value.
       The Numerical pipeline consists of a SimpleImputer() and a StandardScaler(). The Ordinal pipeline utilises 
       a SimpleImputer() and a OrdinalEncoder(). The Nominal Pipeline consists of a SimpleImputer() and a CountEncoder(). 
       The Boolean pipeline also utilizes a StandardImputer, and encoded False as 0 and True as 1.
       CountEncoder was picked in order to prevent data leakage, and in order to make the pipeline more flexible when switching between
       regression and classification tasks. All of the pipelines are combined in a ColumnTransformer() which is called on the entire dataset.

    Args:
        X (_type_): The X matrix of the training set
        numerical_cols (_type_): List of numerical columns in the training set
        ordinal_cols (_type_): List of ordinal columns in the training set
        ordinal_categories (_type_): List of categories for each ordinal variable (i.e [low, medium, high])
        nominal_cols (_type_): List of nominal columns in the training set
        bool_cols (_type_): List of Boolean columns

    Returns:
        tuple[ColumnTransformer, ndarray]: fitted preprocessing pipeline and processed training set.
    """
    # Numerical pipeline: impute + scale
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Ordinal pipeline: impute + ordinal encode
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_categories,
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])
    
    # Nominal pipeline: impute + target encode
    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', ce.CountEncoder())
    ])

    bool_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # optional
    ('to_int', FunctionTransformer(to_int_pre, feature_names_out="one-to-one"))
    ])
    
    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
        ('bool', bool_pipeline, bool_cols),
        ('nom', nominal_pipeline, nominal_cols)
    ])
    final_preprocessor = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])

    #X_processed = preprocessor.fit_transform(X)
    X_processed = final_preprocessor.fit_transform(X)
    return final_preprocessor, X_processed

def prepreprocessing_pipeline_for_clustering(X, numerical_cols, ordinal_cols, ordinal_categories, nominal_cols, bool_cols):
    """Creates a flexible preprocessing pipeline meant to handle numerical, ordinal, nominal and boolean values. 
       The pipeline is created using sklearn.pipeline.Pipeline(), where a pipeline is created for each type of value.
       The Numerical pipeline consists of a SimpleImputer() and a StandardScaler(). The Ordinal pipeline utilises 
       a SimpleImputer() and a OrdinalEncoder(). The Nominal Pipeline consists of a SimpleImputer() and a CountEncoder(). 
       The Boolean pipeline also utilizes a StandardImputer, and encoded False as 0 and True as 1.
       CountEncoder was picked in order to prevent data leakage, and in order to make the pipeline more flexible when switching between
       regression and classification tasks. All of the pipelines are combined in a ColumnTransformer() which is called on the entire dataset.

    Args:
        X (_type_): The X matrix of the training set
        numerical_cols (_type_): List of numerical columns in the training set
        ordinal_cols (_type_): List of ordinal columns in the training set
        ordinal_categories (_type_): List of categories for each ordinal variable (i.e [low, medium, high])
        nominal_cols (_type_): List of nominal columns in the training set
        bool_cols (_type_): List of Boolean columns

    Returns:
        tuple[ColumnTransformer, ndarray]: fitted preprocessing pipeline and processed training set.
    """
    # Numerical pipeline: impute + scale
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Ordinal pipeline: impute + ordinal encode
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_categories,
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])
    
    # Nominal pipeline: impute + target encode
    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False))
    ])

    bool_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # optional
    ('to_int', FunctionTransformer(to_int_pre, feature_names_out="one-to-one"))
    ])
    
    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
        ('bool', bool_pipeline, bool_cols),
        ('nom', nominal_pipeline, nominal_cols)
    ])
    final_preprocessor = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])

    #X_processed = preprocessor.fit_transform(X)
    X_processed = final_preprocessor.fit_transform(X)
    feature_names = final_preprocessor.named_steps['preprocessor'].get_feature_names_out()
    df_out = pd.DataFrame(X_processed, columns=feature_names, index = X.index)
    return final_preprocessor, df_out

def unfit_prepreprocessing_pipeline(numerical_cols, ordinal_cols, ordinal_categories, nominal_cols, bool_cols):
    """Creates a flexible preprocessing pipeline meant to handle numerical, ordinal, nominal and boolean values. 
       The pipeline is created using sklearn.pipeline.Pipeline(), where a pipeline is created for each type of value.
       The Numerical pipeline consists of a SimpleImputer() and a StandardScaler(). The Ordinal pipeline utilises 
       a SimpleImputer() and a OrdinalEncoder(). The Nominal Pipeline consists of a SimpleImputer() and a CountEncoder(). 
       The Boolean pipeline also utilizes a StandardImputer, and encoded False as 0 and True as 1.
       CountEncoder was picked in order to prevent data leakage, and in order to make the pipeline more flexible when switching between
       regression and classification tasks. All of the pipelines are combined in a ColumnTransformer() which is called on the entire dataset.

    Args:
        X (_type_): The X matrix of the training set
        numerical_cols (_type_): List of numerical columns in the training set
        ordinal_cols (_type_): List of ordinal columns in the training set
        ordinal_categories (_type_): List of categories for each ordinal variable (i.e [low, medium, high])
        nominal_cols (_type_): List of nominal columns in the training set
        bool_cols (_type_): List of Boolean columns

    Returns:
        tuple[ColumnTransformer, ndarray]: fitted preprocessing pipeline and processed training set.
    """
    # Numerical pipeline: impute + scale
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Ordinal pipeline: impute + ordinal encode
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_categories,
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])
    
    # Nominal pipeline: impute + target encode
    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', ce.CountEncoder())
    ])

    bool_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # optional
    ('to_int', FunctionTransformer(to_int_pre, feature_names_out="one-to-one"))
    ])
    
    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
        ('bool', bool_pipeline, bool_cols),
        ('nom', nominal_pipeline, nominal_cols)
    ])    
    return preprocessor

def apply_preprocessing_pipeline(X, preprocessor):
    """Applies the preprocessing pipeline to the test and val sets, excluding the target variable

    Args:
        X (_type_): Test or Val set
        preprocessor (_type_): preprocessing ColumnTransformer

    Returns:
        _type_: Processed data
    """
    # Apply the main Column Transformer
    X_processed = preprocessor.transform(X)
    return X_processed

def remove_null(X, y):
    not_null_mask = y.notnull()
    X = X[not_null_mask].copy()
    y = y[not_null_mask].copy()
    return X, y
def get_feature_names_from_column_transformer(ct):
    feature_names = []

    for name, transformer, cols in ct.transformers_:
        if name != 'remainder' and transformer != 'drop':
            # If pipeline, get last step names if available
            if hasattr(transformer, 'named_steps'):
                last_step = list(transformer.named_steps.values())[-1]
                if hasattr(last_step, 'get_feature_names_out'):
                    feature_names.extend(last_step.get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
    return feature_names


