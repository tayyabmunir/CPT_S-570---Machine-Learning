# # Q 2  [Kernelized Perceptron]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


Data=pd.read_csv('C:/Users/tayya/OneDrive/Desktop/Fall 2020/Machine Learning/Assignment 2/Data/fashion-mnist_train.csv')
y_train=np.zeros(len(Data))
X_train=np.zeros([len(Data),784])
for i in range(len(Data)):
    y_train[i]=Data.loc[i,'label']
    X_train[i]=Data.loc[i,'pixel1':'pixel784']


Data_test=pd.read_csv('C:/Users/tayya/OneDrive/Desktop/Fall 2020/Machine Learning/Assignment 2/Data/fashion-mnist_test.csv')
y_test=np.zeros(len(Data_test))
X_test=np.zeros([len(Data_test),784])
for i in range(len(Data_test)):
    y_test[i]=Data_test.loc[i,'label']
    X_test[i]=Data_test.loc[i,'pixel1':'pixel784']


def poly_Kernel(X_train,xj,p):
    K=np.zeros(len(X_train))
    K=(1+np.matmul(X_train,np.reshape(xj,(784,1))))**p
    return K


def Predict_Y(Alpha_M,x,X_train,k,p):
    pred_vect=np.zeros(k)
    pred_vect=np.matmul(Alpha_M,poly_Kernel(X_train,x,p))
    pred=np.argmax(pred_vect)
    return pred                   


def Train_Kernelized_Perceptron(X_train,y_train,k,T,p):
    Alpha_M=np.zeros([k,len(X_train)])
    Errors=np.array([])
    for i in range(T):
        Count=0
        for j in range(len(X_train)):
            t0 =time.time()
            yhat=Predict_Y(Alpha_M,X_train[j],X_train,k,p)
            t1= time.time()   
            if(yhat!=y_train[j]):
                Alpha_M[int(yhat),j]=Alpha_M[int(yhat),j]-1
                Alpha_M[int(y_train[j]),j]=Alpha_M[int(y_train[j]),j]+1  
                Count=Count+1
                if(j%1000==0):
                    print(t1-t0,j) 
                #print(j)            
        Errors=np.append(Errors,Count)
        print("Pass Comp")
    return Alpha_M, Errors


# Training


from sklearn import preprocessing
X_train=preprocessing.scale(X_train)
X_test=preprocessing.scale(X_test)


X_train_set=X_train[0:48000]
y_train_set=y_train[0:48000]
X_train_val=X_train[48000:60000]
y_train_val=y_train[48000:60000]



[Alpha_M,Errors]=Train_Kernelized_Perceptron(X_train_set,y_train_set,10,5,2)


Training_IT=[1,2,3,4,5]
plt.plot(Training_IT,(48000-Errors)/48000) 
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.title('Online Learning Curve-Kernelized Perceptron')
plt.show()


Training_IT=[1,2,3,4,5]
plt.plot(Training_IT,Errors) 
plt.xlabel('Training Iterations')
plt.ylabel('Mistakes')
plt.title('Online Learning Curve-Kernelized Perceptron')
plt.show()


# Comparison of the Training, Test and validation Accuracy


Mistake=0
for i in range(len(X_train_val)):
    yhat=Predict_Y(Alpha_M,X_train_val[i],X_train_set,10,2)
    print(i)
    if(yhat!=y_train_val[i]):
        Mistake=Mistake+1


Mistake_test=0
for i in range(len(X_test)):
    yhat=Predict_Y(Alpha_M,X_test[i],X_train_set,10,2)
    print(i)
    if(yhat!=y_test[i]):
        Mistake_test=Mistake_test+1


print("Training Accuracy   :",(48000-Errors[4])/48000)
print("Validation Accuracy :",(12000-Mistake)/12000)
print("Test Accuracy       :",(10000-Mistake_test)/10000)

