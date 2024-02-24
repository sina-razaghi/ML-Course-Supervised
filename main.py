import numpy as np
import pandas as pd

data = pd.read_csv("D:\Lessons\Machine Learcning\Training\T2\Bodyfat.csv")

data = data.fillna(method='ffill')

x = data.iloc[:, 1:10].values
y = data.iloc[:, 0:1].values

print(data.head())
print(np.shape(data))
print(x[0], y[0])

print(data.describe())

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

# process = PolynomialFeatures(degree=4) #instance of PolynomialFeatures
# converting = process.fit_transform(x)
# print(f"orginal sample : {x[0]} \nMapped to {converting[0]}")

# regression = LinearRegression()
# regression.fit(converting, y)
# y_estimate = regression.predict(converting)
# print(mean_squared_error(y, y_estimate, squared=False))
# print(regression.coef_)

def Polynominal(x_Train, y_Train, m):
    poly = PolynomialFeatures(degree=m)

import matplotlib.pyplot as plt

numbers = [1 , 2 , 3, 4 , 5]
emptyList = []

for i in numbers:
    print(f"i = {i}")
    poly = PolynomialFeatures(degree=i)
    X_poly = poly.fit_transform(x)
    regression = LinearRegression()
    regression.fit(X_poly, y)
    mse = cross_val_score(regression, x, y, cv=10)
    print(mse.mean())
    emptyList.append(mse.mean())

plt.scatter(numbers, emptyList, color='red')
plt.plot(numbers, emptyList, color='blue')
plt.show()


import warnings
warnings.filterwarnings(action='ignore')



