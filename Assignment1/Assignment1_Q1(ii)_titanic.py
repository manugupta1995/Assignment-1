
# coding: utf-8

# In[323]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

train = pd.read_csv('/home/rajarshi/Downloads/titanic/train.csv')
test = pd.read_csv('/home/rajarshi/Downloads/titanic/test.csv')
train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1)
test = test.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train = train.fillna(train.mean())
test = test.fillna(test.mean())
y = train['Survived']
y = y.reshape(y.shape[0],1)
x = train.drop(['Survived'],axis = 1)
#print x_train.shape , x_test.shape , y_train.shape, y_test.shape
#print x_train.isnull().any()
x_train , x_test, y_train , y_test = train_test_split(x , y, train_size = .8, test_size = 0.2)
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
m = x_train.shape[1]
n = x_test.shape[1]
w = np.zeros((x_train.shape[0],1))
b = 0
alpha = 0.003
#m = x_train.shape[0]
#z = np.dot(x_train, theta) + b 
#h_theta = sigmoid(z)
N = 60000
cost = np.zeros((N,1))
for i in range(N):
    z = np.dot(w.T,x_train) + b
    #print z.shape 
    h_theta = sigmoid(z)
    dz = h_theta - y_train
    #print dz.shape
    db = np.sum(dz)/m
    dw = np.dot(x_train,dz.T)/m
    #print w.shape , dw.shape
    w = w - alpha*dw
    b = b - alpha*db
    cost[i] = - np.sum( np.multiply(y_train,np.log(h_theta)) + np.multiply(1-y_train, np.log(1 - h_theta)) )/m
    #df = [i , cost1] 
    #print "df =" ,costi
    #cost = cost.append(costi)
    #print cost
#print cost
#print w.shape , b , cost #, z.shape
plt.plot(cost)
#fig.suptitle('test title')
plt.xlabel('number of iteration')
plt.ylabel('cost')
z = np.dot(w.T,x_test) + b
    #print z.shape 
h_theta = sigmoid(z)
cost = - np.sum( np.multiply(y_test,np.log(h_theta)) + np.multiply(1-y_test, np.log(1 - h_theta)) )/n
print "cost = " ,cost
h_theta = np.floor(h_theta*2)
print "Accuracy = " , float(np.sum(h_theta == y_test))/y_test.shape[1]

