from sklearn.linear_model import TweedieRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import seaborn as sns

def find_dist(y):
    ser_y = pd.Series(y, name="Target Vector dist")
    #Plot data
    fig, ax = plt.subplots()
    sns.displot(y, bins=100, color="g", ax=ax)
    plt.show()
    pass

def find_dist_h(y):
    ser_y = pd.Series(y, name="Target Vector dist")
    #Plot data
    fig, ax = plt.subplots()
    sns.histplot(y, bins=100, color="g", ax=ax)
    plt.show()
    pass

def tweed(X_train, y_train, params):
    tweed_model = TweedieRegressor(power=params["power"], alpha=params["alpha"], link='log', max_iter=1000)
    #Fit model
    tweed_model.fit(X_train, y_train)
    return tweed_model
def apply_tweed(X, y, tweed):
    y_pred = tweed.predict(X)
    msa = mean_absolute_error(y, y_pred)
    mspa = mean_absolute_percentage_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    return [msa, mspa, mse, rmse], y_pred

def optimize(X,y, X_val, y_val):
    best_mae = float("inf")
    best_params = None
    best_model = None

    powers = [x for x in np.arange(1.0, 2.01, 0.1)]
    alphas = [x for x in np.arange(0, 1.0, 0.1)]

    for p in powers:
        for a in alphas:
            model = TweedieRegressor(power=p, alpha=a, link='log', max_iter=5000)
            model.fit(X, y.squeeze())
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            if mae < best_mae:
                best_model = model
                best_params = {'power':p, 'alpha':a}
                best_mae = mae
    return {'model':best_model, 'params':best_params, 'MAE':best_mae}

def CV_optimize(X, y):
    model = TweedieRegressor(link='log',max_iter=100000)
    parameters = {'power':[x for x in np.arange(1.0, 2.0, 0.1)], 
                  'alpha':[x for x in np.arange(0, 1.0, 0.1)]}
    cv = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', refit=True)
    cv.fit(X,y.squeeze())
    best_params = cv.get_params()
    best_model = cv.best_estimator_
    best_score = -cv.best_score_
    return best_model, best_params, best_score