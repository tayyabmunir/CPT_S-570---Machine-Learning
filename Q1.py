#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import linalg, optimize
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from sklearn import svm


# # Loading Data 

# In[2]:


file = open('./data/traindata.txt', 'r') 
FC_Lines_Train = file.readlines()
FC_Lines_Train=[x.strip() for x in FC_Lines_Train]
file.close()
file = open('./data/stoplist.txt', 'r') 
Stop=file.readlines()    
Stop = [x.strip() for x in Stop] 
file.close()
file = open('./data/testdata.txt', 'r') 
FC_Lines_Test = file.readlines()
FC_Lines_Test=[x.strip() for x in FC_Lines_Test]
file.close()
file = open('./data/testlabels.txt', 'r') 
y_test = [[int(v) for v in line.split()] for line in file]
y_test=np.array(y_test)
file.close()
y_test=np.array(y_test)
file = open('./data/trainlabels.txt', 'r') 
y_train = [[int(v) for v in line.split()] for line in file]
y_train=np.array(y_train)
file.close()
y_train=np.array(y_train)


# # Generating Vocabulary and Features

# In[3]:


def Generate_Vocabulary(FC_Lines,Stop):
    W=[]
    for line in FC_Lines:
        W.extend(line.strip().split(' '))
    W=sorted(W)
    Vocabulary=[]
    for i in range(len(W)):
        if W[i] not in Stop:
            if W[i] not in Vocabulary:
                Vocabulary.append(W[i])
    Count=[]
    for i in range(len(Vocabulary)):
        CNT=0
        for j in range(len(W)):
            if(Vocabulary[i]==W[j]):
                CNT=CNT+1
        Count.append(CNT)
    return Vocabulary,Count
def Generate_Features(FC_Lines,Vocabulary):
    X_train=np.zeros([len(FC_Lines),len(Vocabulary)])
    for i in range(len(FC_Lines)):
        Temp=FC_Lines[i].strip().split(' ')
        for k in range(len(Vocabulary)):
            if Vocabulary[k] in Temp:
                X_train[i,k]=1
    return X_train


# In[4]:


[Vocabulary,Count]=Generate_Vocabulary(FC_Lines_Train,Stop)


# In[6]:


X_train=Generate_Features(FC_Lines_Train,Vocabulary)
X_test=Generate_Features(FC_Lines_Test,Vocabulary)


# # Training Naive Bayes Classifier: Calculating Naive Bayes Probabilities

# In[7]:


def Probabilities(y_train,X_train):
    Count_Yes=0
    Count_Pos_1=np.zeros([len(X_train[0]),1])
    Count_Pos_0=np.zeros([len(X_train[0]),1])
    for i in range(len(y_train)):
        if((y_train[i][0])==1):
            Count_Yes+=1
            for k in range(len(X_train[i])):
                if((X_train[i][k])==1):
                    Count_Pos_1[k]+=1
        else:
            for k in range(len(X_train[i])):
                if((X_train[i][k])==1):
                    Count_Pos_0[k]+=1        
    Pr_Yes=Count_Yes/len(y_train)
    Pr_No=1-Pr_Yes
    Count_No=len(y_train)-Count_Yes
    Pos_prob=(Count_Pos_1+1)/(Count_Yes+2)
    Neg_prob=(Count_Pos_0+1)/(Count_No+2)
    CP=(Count_Pos_1+Count_Pos_0)
    PXi=(CP+1)/(len(y_train)+2)
    return Pr_Yes,Pr_No,Pos_prob,Neg_prob,PXi 


# In[8]:


[Pr_Yes,Pr_No,Pos_prob,Neg_prob,PXi]=Probabilities(y_train,X_train)


# In[12]:


print(Neg_prob)


# # Predict Function

# In[14]:


def predict(X_testk,Pr_Yes,Pr_No,Pos_prob,Neg_prob,PXi):
    Pr_XP=Pr_Yes
    Pr_XN=Pr_No
    PX=1
    for i in range(len(X_testk)):
        if((X_testk[i])==1):
            Pr_XP=Pr_XP*Pos_prob[i]
            Pr_XN=Pr_XN*Neg_prob[i]
            PX=PX*PXi[i]
        else:
            Pr_XP=Pr_XP*(1-Pos_prob[i])
            Pr_XN=Pr_XN*(1-Neg_prob[i])
            PX=PX*(1-PXi[i])
    P_1=Pr_XP/PX
    P_0=Pr_XN/PX
    if(P_1>=P_0):
        y_pred=1
    else:
        y_pred=0
    return y_pred


# In[ ]:





# # Accuracy for Training and Testing Data

# In[15]:


Rights=0
for i in range(len(y_test)):
    ypred=predict(X_test[i],Pr_Yes,Pr_No,Pos_prob,Neg_prob,PXi)
    if (ypred==y_test[i][0]):
        Rights=Rights+1
Testing_Accuracy=Rights/len(y_test)
Rights=0
for i in range(len(y_train)):
    ypred=predict(X_train[i],Pr_Yes,Pr_No,Pos_prob,Neg_prob,PXi)
    if (ypred==y_train[i][0]):
        Rights=Rights+1
Training_Accuracy=Rights/len(y_train)
print('Test Accuracy:',Testing_Accuracy,'\nTrain Accuracy:',Training_Accuracy)

