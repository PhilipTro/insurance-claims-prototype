import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from pathlib import Path
import os
import sys
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path.home() / "desktop" / "Insurance_Project"


def load_data(switch = True):
    if switch:
        X_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "classification"/ "X_train.csv")
        X_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "classification"/ "X_test.csv")
        y_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "classification"/ "y_train.csv")
        y_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "classification"/ "y_test.csv")
        return X_train, X_test, y_train, y_test
    
def train_classifier(X, y, models=None, param_grid=None):
    y = y.squeeze()
    scoring = {
        'roc_auc': 'roc_auc',
        'f1': 'f1'
    }

    if models==None:
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        models = {
            'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
        }
    if param_grid==None:
        param_grid = {
            'RandomForest': {
                'n_estimators': [200, 500],
                'max_depth': [None, 10, 20]
            },
            'XGBoost': {
                'n_estimators': [200, 500],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        }
    results = {}
    for name, model in models.items():
        print(f"training {name}...")
        if name in param_grid:
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid[name],
                scoring=scoring,
                refit='f1',
                cv=5,
                n_jobs=1,
                verbose=1
            )
            
            grid.fit(X, y)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            model.fit(X,y)
            best_model = model
            best_params = None
        #Evaluate
        y_pred = best_model.predict(X)
        y_prob = best_model.predict_proba(X)[:,1]
        results[name] = {
            'model':best_model,
            'best_params':best_params,
            'roc_auc':roc_auc_score(y, y_prob),
            'f1_score':f1_score(y,y_pred),
            'classification_report':classification_report(y,y_pred)
        }

    return results


def evaluate_classifer(X, y , raw_models:dict):
    results = {}
    models = {}
    for name, dict in raw_models.items():
        models[name] = dict['model']

    for name, model in models.items():
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:,1]
        results[name] = {
            'roc_auc':roc_auc_score(y, y_prob),
            'f1_score':f1_score(y,y_pred),
            'classification_report':classification_report(y,y_pred),
            'confusion_matrix':confusion_matrix(y,y_pred)
        }        
    
    return results