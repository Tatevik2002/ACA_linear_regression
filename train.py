import numpy as np
import pandas as pd
from linreg import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def fit_methods(methods, x, y):
    for method in methods:
        model = LinearRegression(method,x, y)
        weights = model.fit(x, y)
        print(model.loss_function())
        print(weights)


df = pd.read_csv('ecommerce.csv')
df = df.iloc[:, 3:]
x = df['Length of Membership'].to_numpy()
y = df['Yearly Amount Spent'].to_numpy()
x= x.reshape(-1,1)
y = y.reshape(-1,1)

fit_methods(x,y, ('analytic'))

# TODO: compare your results with the same models on sklearn
# LinearRegression, SGDRegressor - sklearn objects
def compair(method,x,y):
    if method == "analytic":
        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        reg = LinearRegression().fit(x, y)
        print(reg.coef_)
    if method == "sgd":
        y = y.ravel()
        reg =  SGDRegressor(max_iter=1000, tol=1e-3)
        reg.fit(x, y)

        print(reg.coef_)
    
compair("sgd",x, y)


