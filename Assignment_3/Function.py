import numpy as np
import matplotlib.pyplot as plt

def trans_to_one(input):
    output = np.ones((input.size, input.max()+1)) * (-1)
    output[np.arange(input.size), input] = 1
    return output


def reverce_ones(input):
    output = []
    for i in input: output.append(int(np.where(i == 1.0)[0])+1)
    return output


def predict(x, y, w):
    output = np.zeros(x.shape[0])
    for a in range(x.shape[0]):
        for b in range(y.shape[1]):
            y_pre = np.dot(w[:,b],x[a,:]) 
            if y_pre > 0: output[a] = b; break
    return output


def errors(y, predict):
    e = 0
    n = y.shape[0]
    for i in range(n):
        target = y[i,:]
        if target[int(predict[i])]!=1.0: e+=1
    return e/n


def sign(x):
    if x >= 0: return 1
    else : return -1


def perceptron(X_train, Y_train, epochs, Y_train_plot, number, lastWeights):
    # ******************************************* Start Perceptron *******************************************
    epoch = 1
    misscalificatino = 1
    w = np.zeros((X_train.shape[1],1))   

    while(misscalificatino !=0 and epoch <= epochs):      
        misscalificatino = 0 
        for xi,yi in zip(X_train,Y_train): 
            y_hat = sign(np.dot(w.T,xi)[0])                                     
            if yi*y_hat < 0:                 
                w = (w.T + yi*xi).T         
                misscalificatino = misscalificatino + 1  
    
        epoch = epoch + 1      
    
    # ******************************************* Draw weights *******************************************
    # draw(X_train, Y_train_plot, number, w, lastWeights)
    
    return w


def draw(X, Y, number, w, lastWeights):
    # ******************************************* Plot Variables *******************************************
    colors = ["black", "blue", "green", "orange"]
    plt.axis([min(X[:,0])*2 , max(X[:,0])*2 , min(X[:,1])*2 , max(X[:,1])*2])
    x1 = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)
    x2 = x1  
    
    # ******************************************* Draw *******************************************
    plt.clf()
    plt.scatter(X[:,0], X[:,1], marker="o", c=Y, s=25, edgecolor="k")
    plt.plot(x1, -1*(w[0][0]*x2  + w[2][0])/w[1][0] ,color=colors[number])
    plt.title(f'Class {number+1}')
    try:
        plt.plot(x1, -1*(lastWeights[0][0]*x2  + lastWeights[2][0])/lastWeights[1][0] ,color=colors[0] ,label='Class 1')
        plt.plot(x1, -1*(lastWeights[0][1]*x2  + lastWeights[2][1])/lastWeights[1][1] ,color=colors[1] ,label='Class 2')
        plt.plot(x1, -1*(lastWeights[0][2]*x2  + lastWeights[2][2])/lastWeights[1][2] ,color=colors[2] ,label='Class 3')
    except:
        pass
    plt.pause(2)  