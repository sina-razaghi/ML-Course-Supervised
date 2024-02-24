import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("example1.csv")
X = dataset.iloc[:, 0:2].values
t = dataset.iloc[:, 2:3].values
Y = np.array([(i[0]) for i in t])

plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")
plt.show()

bias = np.ones((len(X), 1))
X = np.hstack((X, bias))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

print(f"Train Sampels => {np.shape(X_train)[0]}")
print(f"Test Sampels => {np.shape(X_test)[0]}")
print(f"{np.shape(Y_test)}")

Y_train_ = Y_train.copy()
o = np.ones((Y_train.size, Y_train.max()+1)) * (-1)
o[np.arange(Y_train.size), Y_train] = 1
Y_train = o


weights = np.zeros((X_train.shape[1], 1))


def signum(x):
    if x >=0: return 1
    else : return -1


def Perceptron(X_train, Y_train, K, epochs):
    epoch = 1
    m = 1
    w = np.zeros((3 , 1))

    while(m!=0 and epoch <= epochs):      
        m = 0 

        for xi,yi in zip(X_train,Y_train): 

            y_hat = signum(np.dot(w.T,xi)[0])                                     
            if yi*y_hat < 0:          
                w = (w.T + yi*xi).T         
                m = m + 1     

        epoch = epoch + 1     

    return w

x1 = np.arange(min(X_train[:, 0]), max(X_train[:, 0]), 0.01)
x2 = x1  

all_weights = []
for round in range(0,4):
    w = Perceptron(X_train, Y_train[:,round], round, 100)
    all_weights.append(w)
    print(f"w => {w}")
    plt.clf()
    plt.scatter(X_train[:,0], X_train[:,1], marker="o", c=Y_train_, s=25, edgecolor="k")
    plt.plot(x1, -1*(w[0][0]*x2  + w[2][0])/w[1][0])
    plt.show()

print(f"all a => {all_weights}")

plt.clf()
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=Y_train_, s=25, edgecolor="k")
x1 = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)
x2 = x1
plt.plot(x1, -1*(all_weights[0][0]*x2  + all_weights[0][2])/all_weights[0][1] ,color="red" ,label='1')
plt.plot(x1, -1*(all_weights[1][0]*x2  + all_weights[1][2])/all_weights[1][1] ,color="blue" ,label='2')
plt.plot(x1, -1*(all_weights[2][0]*x2  + all_weights[2][2])/all_weights[2][1] ,color="green" ,label='3')
plt.plot(x1, -1*(all_weights[3][0]*x2  + all_weights[3][2])/all_weights[3][1] ,color="brown" ,label='4')
plt.title("Training Data")
plt.show()


plt.clf()
plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=Y_test, s=25, edgecolor="k")
x1 = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)
x2 = x1
plt.plot(x1, -1*(all_weights[0][0]*x2  + all_weights[0][2])/all_weights[0][1] ,color="red" ,label='1')
plt.plot(x1, -1*(all_weights[1][0]*x2  + all_weights[1][2])/all_weights[1][1] ,color="blue" ,label='2')
plt.plot(x1, -1*(all_weights[2][0]*x2  + all_weights[2][2])/all_weights[2][1] ,color="green" ,label='3')
plt.plot(x1, -1*(all_weights[3][0]*x2  + all_weights[3][2])/all_weights[3][1] ,color="brown" ,label='4')
plt.title("Testing Data")
plt.show()

plt.clf()
plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")
x1 = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)
x2 = x1
plt.plot(x1, -1*(all_weights[0][0]*x2  + all_weights[0][2])/all_weights[0][1] ,color="red" ,label='1')
plt.plot(x1, -1*(all_weights[1][0]*x2  + all_weights[1][2])/all_weights[1][1] ,color="blue" ,label='2')
plt.plot(x1, -1*(all_weights[2][0]*x2  + all_weights[2][2])/all_weights[2][1] ,color="green" ,label='3')
plt.plot(x1, -1*(all_weights[3][0]*x2  + all_weights[3][2])/all_weights[3][1] ,color="brown" ,label='4')
plt.title("All Data")
plt.show()
