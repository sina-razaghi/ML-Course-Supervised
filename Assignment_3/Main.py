import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Function

# ******************************************* Enter Dataset *******************************************

dataset = pd.read_csv("example.csv")
X = dataset.iloc[:, 0:2].values
t = dataset.iloc[:, 2:3].values
Y = np.array([(i[0]) for i in t])

plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")
plt.show()

colors = ["black", "blue", "green", "orange"]

# ******************************************* Split Dataset *******************************************

Y_all_plot = Y.copy()

Y = Function.trans_to_one(Y)

bias = np.ones((len(X), 1))
X = np.hstack((X, bias))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

print(f"Train Sampels => {np.shape(X_train)[0]}")
print(f"Test Sampels => {np.shape(X_test)[0]}")

Y_train_plot = np.array(Function.reverce_ones(Y_train))
Y_test_plot = np.array(Function.reverce_ones(Y_test))

# **************************************** Start Calculate with K-Fold ****************************************

from sklearn.model_selection import KFold
kf = KFold(n_splits = 16)
kf.get_n_splits(X_train)

count_fold = 0
weights = []
final_weights = []
accuracies = []

for train_index, test_index in kf.split(X_train):
    count_fold += 1
    print(f"\nkfold => {count_fold}")
    print("Train:", np.shape(train_index)[0], "/ Test:", np.shape(test_index)[0])
    local_weights = np.zeros((X_train.shape[1], Y_train.shape[1]))
    x_ = X_train[train_index]
    y_ = Y_train[train_index]
    y_plot = Y_train_plot[train_index]
    for i in range(y_.shape[1]):
        w = Function.perceptron(x_, y_[:,i], 100, y_plot, i, local_weights)
        local_weights[:,i] = w[:,0]

    x__ = X_train[test_index]
    y__ = Y_train[test_index]
    predictedclass = Function.predict(x__, y__, local_weights)
    ac = 100 - round((Function.errors(y__, predictedclass)*100),2)
    print(f"accuracy => {ac}%")
    accuracies.append(ac)
    weights.append(local_weights)

print(f"\nWeights => \n{weights}")

# ****************************** Get Average from Weights, learn Accuracy and Test Accuracy ******************************

weights = np.array(weights)
avg = 0
avgs = []

for a in range(len(weights[0])):
    for b in range(len(weights[0][a])):
        avg = np.average(weights[:, a, b])
        avgs.append(avg)
    final_weights.append(np.array(avgs))
    avg = 0
    avgs = []

print(f"\nFinal Weights => \n{final_weights}")
final_weights = np.array(final_weights)

predictedclass = Function.predict(X_test, Y_test, final_weights)
test_ac = 100 - round((Function.errors(Y_test, predictedclass)*100),2)

avg_train_ac = sum(accuracies) / len(accuracies)

# ******************************************* Show Final Results *******************************************

print(f'''
============= Result =============\n
C1 weights = {np.round(final_weights[:,0],2)}
C2 weights = {np.round(final_weights[:,1],2)}
C3 weights = {np.round(final_weights[:,2],2)}
C4 weights = {np.round(final_weights[:,3],2)}

Avrage Train Accuracy = {np.round(avg_train_ac, 2)}%

Test Accuracy = {np.round(test_ac, 2)}%
''')

# ******************************************* Draw final weights on train data *******************************************

plt.clf()
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=Y_train_plot, s=25, edgecolor="k")
x1 = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)
x2 = x1
plt.plot(x1, -1*(final_weights[0][0]*x2  + final_weights[2][0])/final_weights[1][0] ,color=colors[0] ,label='1')
plt.plot(x1, -1*(final_weights[0][1]*x2  + final_weights[2][1])/final_weights[1][1] ,color=colors[1] ,label='2')
plt.plot(x1, -1*(final_weights[0][2]*x2  + final_weights[2][2])/final_weights[1][2] ,color=colors[2] ,label='3')
plt.plot(x1, -1*(final_weights[0][3]*x2  + final_weights[2][3])/final_weights[1][3] ,color=colors[3] ,label='4')
plt.title("---------- Train Sampels ----------")
plt.show()

# ******************************************* Draw final weights on test data *******************************************

plt.clf()
plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=Y_test_plot, s=25, edgecolor="k")
x1 = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)
x2 = x1
plt.plot(x1, -1*(final_weights[0][0]*x2  + final_weights[2][0])/final_weights[1][0] ,color=colors[0] ,label='1')
plt.plot(x1, -1*(final_weights[0][1]*x2  + final_weights[2][1])/final_weights[1][1] ,color=colors[1] ,label='2')
plt.plot(x1, -1*(final_weights[0][2]*x2  + final_weights[2][2])/final_weights[1][2] ,color=colors[2] ,label='3')
plt.plot(x1, -1*(final_weights[0][3]*x2  + final_weights[2][3])/final_weights[1][3] ,color=colors[3] ,label='4')
plt.title("---------- Test Sampels ----------")
plt.show()

