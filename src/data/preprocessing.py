from pathlib import Path
import pandas as pd
import numpy
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

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



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce

def prepreprocessing_pipeline(X, numerical_cols, ordinal_cols, ordinal_categories, nominal_cols, bool_cols):
    """
    Returns a fitted preprocessor and the transformed data.
    
    Parameters
    ----------
    X : pd.DataFrame 
        Feature dataframe
    numerical_cols : list
        List of numerical columns
    ordinal_cols : list
        List of ordinal categorical columns
    ordinal_categories : list of lists
        Ordered categories for each ordinal column
    nominal_cols : list
        List of nominal categorical columns
    
    Returns
    -------
    preprocessor : ColumnTransformer
        Fitted preprocessor
    X_processed : np.ndarray
        Transformed features
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
    ('to_int', FunctionTransformer(lambda x: x.astype(int)))
    ])
    
    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
        ('bool', bool_pipeline, bool_cols),
        ('nom', nominal_pipeline, nominal_cols)
    ])

    #X_nom = nominal_pipeline.fit_transform(X[nominal_cols])
    #X2 = X.copy()
    #X2[nominal_cols] = X_nom

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    return preprocessor, X_processed

# def apply_preprocessing_pipeline(X, preprocessor, nominal_pipeline, nominal_cols, global_mean):
#     # Tranform nominal values using TargetEncoder
#     X_nom = nominal_pipeline.transform(X[nominal_cols])
#     # Replace nominal columns with encoded values
#     X2 = X.copy()
#     X2[nominal_cols] = X_nom
#     X2[nominal_cols] = X2[nominal_cols].fillna(global_mean)
#     # Apply the main Column Transformer
#     X_processed = preprocessor.transform(X2)
#     return X_processed

def apply_preprocessing_pipeline(X, preprocessor):
    # Apply the main Column Transformer
    X_processed = preprocessor.transform(X)
    return X_processed

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


