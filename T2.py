import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt




# Dataset = pd.read_csv("D:\Lessons\Machine Learcning\Training\T2\wind_dataset.csv")
Dataset = pd.read_csv("D:\Lessons\Machine Learcning\Training\T2\X_train.csv")

# Dataset.plot(x='DATE', y='WIND')
# plt.show()

# import seaborn as sns
# sns.heatmap(Dataset.corr(),annot=True, cbar=False, cmap='Blues', fmt='.1f')
# plt.show()

# sns.boxplot(Dataset['WIND'])
# plt.show()
# sns.boxplot(Dataset['RAIN'])
# plt.show()
# sns.boxplot(Dataset['T.MAX'])
# plt.show()
# sns.boxplot(Dataset['T.MIN'])
# plt.show()
# sns.boxplot(Dataset['T.MIN.G'])
# plt.show()
# sns.histplot(Dataset['IND'])
# plt.show()
# sns.histplot(Dataset['IND.1'])
# plt.show()
# sns.histplot(Dataset['IND.2'])
# plt.show()

# from matplotlib import pyplot
# values = Dataset.values
# groups = [0, 1, 2, 3,4, 5, 6, 7]
# i = 1
# pyplot.figure(figsize=(15,15))
# for group in groups:
#     pyplot.subplot(len(groups), 1, i)
#     pyplot.plot(values[:, group])
#     pyplot.title(Dataset.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()


print('Dataset =>\n', Dataset.head())
print('====================================================')
print('Dataset info =>\n')
print(Dataset.info()) #Some info about our attributes and its datatype
print('====================================================')
print('Dataset shape =>\n\n',np.shape(Dataset)) # Dimension of the dataset # What is the shape of the dataset?
print('====================================================')
print('Dataset describe =>\n\n',Dataset.describe()) #Some analysis on the numerical columns
print('====================================================')
print('Dataset null var =>\n\n',Dataset.isnull().sum()) #Check for null values
print('====================================================')
print('Dataset duplicate =>\n\n',Dataset.duplicated().sum()) #Check for duplicates in the dataset
print('====================================================')

# Dataset.hist(figsize=(15,5),grid=False,color='red',bins=15)
# plt.show()

# ============================= missing value =============================
# # Replace the missing values for numerical columns with 0
# df['T.MAX'] = df['T.MAX'].fillna(0)
# df['T.MIN'] = df['T.MIN'].fillna(0)
# df['T.MIN.G'] = df['T.MIN.G'].fillna(0)

# # Replace the missing values for numerical columns with mean
# Dataset['T.MAX'] = Dataset['T.MAX'].fillna(Dataset['T.MAX'].mean())
# Dataset['T.MIN'] = Dataset['T.MIN'].fillna(Dataset['T.MIN'].mean())
# Dataset['T.MIN.G'] = Dataset['T.MIN.G'].fillna(Dataset['T.MIN.G'].mean())

# # Deleting the entire row
# Dataset = Dataset.dropna(axis=0)
# df=df.fillna(0)

print('Dataset null var =>\n\n',Dataset.isnull().sum()) #Check for null values

# X = Dataset.iloc[:, 2:9].values
# Y = Dataset.iloc[:, 1:2].values

X = Dataset.iloc[:, 3:7].values
Y = Dataset.iloc[:, 7:8].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

print('X_training =>\n\n', X_train)
print(np.shape(X_train), np.shape(X_test))
print('====================================================')
print('X_test =>\n\n', X_test)
print('====================================================')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


Model = LinearRegression()
Model.fit(X_train, Y_train)
SC = cross_val_score(Model,X_train, Y_train, scoring="neg_mean_squared_error", cv=10)
print("MSE linear regression", -SC.mean())
Y_pred = Model.predict(X_test)
mse = mean_squared_error(Y_test,Y_pred)
print("MSE prediction linear regression", mse)

def Reg_fit_Poly(X, Y, M=1):
    polynomial_features = PolynomialFeatures(degree=M)
    linear_regression = LinearRegression()
    model = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    model.fit(X, Y)
    return model

Degrees = [1, 2, 3, 4 , 5]
MSE = []
MSE_Pre = []

for M in Degrees:
    print(f"\nRegression degree = {M}")
    Model = Reg_fit_Poly(X_train, Y_train, M=M)
    SC = cross_val_score(Model,X_train, Y_train, scoring="neg_mean_squared_error", cv=10)
    mse = -SC.mean()
    print("MSE Polynomial Model = ", mse)
    MSE.append(mse)
    Y_pred = Model.predict(X_test)
    mse_pre = mean_squared_error(Y_test,Y_pred)
    print("MSE Prediction Polynomial Model = ", mse_pre)
    MSE_Pre.append(mse_pre)


plt.plot(Degrees, MSE, marker="o", color='blue', label='MSE')
plt.plot(Degrees, MSE_Pre, marker="o", color='red', label='MSE_Pre')
plt.title("")
plt.xlabel("degrees")
plt.ylabel("MSE")
plt.show()

print("\n--------- Next1 ---------\n")

from sklearn.linear_model import Ridge

def Regularization_Reg_fit_Poly(X, Y, degree=1, lmbda=1.0):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = Ridge(alpha=lmbda)
    model = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    model.fit(X, Y)
    return model

L = [1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
M = 3
MSE = []
MSE_Pre = []

for _lambda in L:
    Model = Regularization_Reg_fit_Poly(X_train, Y_train, degree=M, lmbda=_lambda)
    SC = cross_val_score(Model, X_test, Y_test, scoring="neg_mean_squared_error", cv=100)
    mse = -SC.mean()
    print(_lambda,"MSE Ridge Polynomial Model =", mse)
    MSE.append(mse)
    Y_pred = Model.predict(X_test)
    mse_pre = mean_squared_error(Y_test,Y_pred)
    print("MSE Prediction Polynomial Model (Ridge)= ", mse_pre)
    MSE_Pre.append(mse_pre)

plt.plot(L, MSE, marker="o", color='blue', label='MSE')
plt.plot(L, MSE_Pre, marker="o", color='red', label='MSE_Pre')
plt.title("")
plt.xlabel("degrees")
plt.ylabel("MSE")
plt.show()

print("\n=======================Lasso====================\n")

from sklearn.linear_model import Lasso

def Regularization_Laso_fit_Poly(X, Y, degree=1, lmbda=1.0):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = Lasso(alpha=lmbda)
    model = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    model.fit(X, Y)
    return model

L = [1e-10, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
M = 3
MSE = []
MSE_Pre = []

for _lambda in L:
    Model = Regularization_Reg_fit_Poly(X_train, Y_train, degree=M, lmbda=_lambda)
    SC = cross_val_score(Model, X_test, Y_test, scoring="neg_mean_squared_error", cv=100)
    mse = -SC.mean()
    print(_lambda,"MSE Lassso Polynomial Model  =", mse)
    MSE.append(mse)
    Y_pred = Model.predict(X_test)
    mse_pre = mean_squared_error(Y_test,Y_pred)
    print("MSE Prediction Polynomial Model (Lasso)= ", mse_pre)
    MSE_Pre.append(mse_pre)

plt.plot(L, MSE, marker="o", color='blue', label='MSE')
plt.plot(L, MSE_Pre, marker="o", color='red', label='MSE_Pre')
plt.title("")
plt.xlabel("degrees")
plt.ylabel("MSE")
plt.show()