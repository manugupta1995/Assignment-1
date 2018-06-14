
# coding: utf-8

# In[19]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
Data = pd.read_csv("/home/rajarshi/Downloads/Wine_quality/winequality-red.csv",delimiter=';')
y = Data["quality"]
y = y.reshape(y.shape[0],1)
Data= Data.drop(["quality"], axis = 1)
x = Data
x_train , x_test, y_train , y_test = train_test_split(x , y, train_size = .8, test_size = 0.2)
n = x_train.shape[0]
m = x_test.shape[0]
k = x_train.shape[1]   #no_of_feature
w = np.zeros((k,1))
b = 0
alpha = 0.0001
cost = np.zeros((500,1))
for i in range(500):
    y_hat = np.dot(x_train,w) + b 
    cost[i] = np.dot((y_hat - y_train).T,(y_hat - y_train))/n
   # if i%100 == 0:
        #print(cost)
    dy_hat = y_hat - y_train
    dw = np.dot(x_train.T , dy_hat)/n
    db = np.sum(dy_hat)/n
    w = w - alpha*dw 
    b = b - alpha*db

# for train dataset 
y_hat_train = np.dot(x_train,w) + b 
cost_train = np.dot((y_hat_train - y_train).T,(y_hat_train - y_train))/n
# for test dataset 
y_hat_test = np.dot(x_test,w) + b 
cost_test = np.dot((y_hat_test - y_test).T,(y_hat_test - y_test))/m
print "the coefficient matrix is: " , w
print "the bias term is:", b 
print "The cost on train data is", cost_train
print "the cost on test data is " ,cost_test
#print "the r-square for train is", r2_score(y_train,y_hat_train)
#print "the r-square for test is", r2_score(y_test,y_hat_test)
y_hat_train = np.rint(y_hat_train)
y_hat_test = np.rint(y_hat_test) 
#print "the accuracy for the train data is" , accuracy_score(y_train,y_hat_train)
#print "the accuracy for the test data is" ,accuracy_score(y_test,y_hat_test)
plt.plot(cost)
plt.xlabel('number of iteration')
plt.ylabel('cost')

