# Q-3) [Decision Tree]


import pandas as pd
import numpy as np

missing_values=['?']
Data=pd.read_csv('C:/Users/tayya/OneDrive/Desktop/Fall 2020/Machine Learning/Assignment 2/Data/breast-cancer-wisconsin.data.csv',header=None,na_values = missing_values)
Data=Data.replace(np.nan,0)
y_train=np.zeros(len(Data))
X_train=np.zeros([len(Data),9])
for i in range(len(Data)):
    if(Data.loc[i,10]==2):
        y_train[i]=1
    else:
        y_train[i]=0
    X_train[i]=Data.loc[i,1:9]


import math
def Entropy(y_train):
    Count1=0
    Pr1=0;
    Pr2=0;
    for i in range(len(y_train)):
        if y_train[i]==1:
            Count1=Count1+1
    Count0=len(y_train)-Count1
    if(len(y_train)!=0):
        Pr1=(Count1)/len(y_train)
        Pr2=1-Pr1;
    else:
        H=0
    if((Pr1!=0)&(Pr2!=0)):
        H=-Pr1*math.log(Pr1)-Pr2*math.log(Pr2)
    else:
        H=0
    return H


# Part a.

def subset(X_train,y_train,TH,i):
    X_T1=[]
    y_T1=np.array([])
    X_T2=[]
    y_T2=np.array([])
    for k in range(len(X_train)):
        if(X_train[k,i]>=TH):
            X_T1.append(X_train[k,:])
            y_T1=np.append(y_T1,y_train[k])
        if(X_train[k,i]<TH):
            X_T2.append(X_train[k])#np.append(X_T2,X_train[k,i])
            y_T2=np.append(y_T2,y_train[k])
    return X_T1,y_T1,X_T2,y_T2 



def Split(X_train,y_train):
    H=Entropy(y_train)
    H_max=np.zeros(X_train.shape[1])
    T_max=np.zeros(X_train.shape[1])
    for i in range(X_train.shape[1]):
        X_T=np.sort(X_train[:,i])
        H1=np.zeros(len(X_T)-1)
        T1=np.zeros(len(X_T)-1)
        for j in range(len(X_T)-1):
            TH=X_T[j]+(X_T[j+1]-X_T[j])/2
            [X_T1,y_T1,X_T2,y_T2]=subset(X_train,y_train,TH,i)
            #print(len(y_T1),len(y_T2))
            #if((len(y_T1)!=0)|(len(y_T2)!=0)):
            H1[j]=Entropy(y_T1)*len(y_T1)/(len(y_T1)+len(y_T2))+Entropy(y_T2)*len(y_T2)/(len(y_T1)+len(y_T2))
            T1[j]=TH
        H_max[i]=max(H-H1)
        T_max[i]=T1[np.argmax(H-H1)]
    Feat_index=np.argmax(H_max)
    Threshold=T_max[Feat_index]
    return Feat_index,Threshold      


# Part b.


def Train_ID3(X_train,y_train):
    Child_1=[]
    Child_2=[]
    [Feat_index,Threshold]=Split(X_train,y_train)
    [X_T1,y_T1,X_T2,y_T2]=subset(X_train,y_train,Threshold,Feat_index)
    if((Entropy(y_T1)==0)&(len(y_T1)>0)):
        Child_1=[int(np.mean(y_T1))]
    elif(len(y_T1)==0):
        Child_1=[]
    if((Entropy(y_T2)==0)&(len(y_T2)>0)):
        Child_2=[int(np.mean(y_T2))]
    elif(len(y_T2)==0):
        Child_2=[]
    if((Entropy(y_T1))!=0):
        subtree1=Train_ID3(np.array(X_T1),y_T1)
        Child_1.extend(subtree1)
    if((Entropy(y_T2))!=0):
        subtree2=Train_ID3(np.array(X_T2),y_T2)
        Child_2.extend(subtree2)
    print('Training Decision Tree ...')
    return Feat_index,Threshold,Child_1,Child_2


Tree=Train_ID3(X_train[0:490],y_train[0:490])


Tree


def pred(X_test,C):
    F=C[0]
    T=C[1]
    if(X_test[F]>=T):
        if ((C[2]==[1])|(C[2]==[0])):
            y_pred=C[2]
            C=C[2]
        else:
            C=C[2]
            y_pred=pred(X_test,C)
    else:
        if ((C[3]==[1])|(C[3]==[0])):
            y_pred=C[3]
            C=C[3]
        else:
            C=C[3]
            y_pred=pred(X_test,C)
    return y_pred



def Testing(X_test,y_test,Tree):
    Mistakes=0
    for i in range(len(X_test)):
        ypred=pred(X_test[i],Tree)
        if(ypred[0]!=y_test[i]):
            Mistakes=Mistakes+1    
    Accuracy=(len(X_test)-Mistakes)/len(X_test)
    return Accuracy


X_trains=X_train[0:490]
y_trains=y_train[0:490]
Accuracy=Testing(X_trains,y_trains,Tree)
print('Training Accuracy:',Accuracy)

X_trainv=X_train[490:560]
y_trainv=y_train[490:560]
Accuracy=Testing(X_trainv,y_trainv,Tree)
print('Validation Accuracy:',Accuracy)

X_test=X_train[560:699]
y_test=y_train[560:699]
Accuracy=Testing(X_test,y_test,Tree)
print('Test Accuracy:',Accuracy)


# Part c.

def Majority(Tree,X_train,y_train):
    [X_T1,y_T1,X_T2,y_T2]=subset(X_train,y_train,Tree[1],Tree[0])
    One=0
    Zero=0
    for i in range(len(y_T1)):
        if(y_T1[i]==0):
            Zero=Zero+1
        else:
            One=One+1
    if(Zero>=One):
        Mp1=0
    else:
        Mp1=1
    One=0
    Zero=0
    for i in range(len(y_T2)):
        if(y_T2[i]==0):
            Zero=Zero+1
        else:
            One=One+1
    if(Zero>=One):
        Mp2=0
    else:
        Mp2=1
    return [Mp1],[Mp2]



def New_Tree(Tree,Mp1,Mp2):
    Tr_P=[Tree[0],Tree[1],Mp1,Mp2]
    return Tr_P



def Prune_Tree(Tree,X_val,y_val):
    T=[]
    [Mp1,Mp2]=Majority(Tree,X_train,y_train)
    Tr_P1=New_Tree(Tree,Mp1,Tree[3])
    Accuracy_p=Testing(X_val,y_val,Tr_P1)
    Accuracy=Testing(X_val,y_val,Tree)
    if(Accuracy_p>Accuracy):
        T.extend(Tr_P1)
    else:
        if((Tree[2]!=[0])&(Tree[2]!=[1])):
            Tr_P1=Prune_Tree(Tree[2],X_val,y_val)
    Tr_P1=New_Tree(Tree,Tree[2],Mp2)
    Accuracy_p=Testing(X_val,y_val,Tr_P1)
    Accuracy=Testing(X_val,y_val,Tree)
    if(Accuracy_p>Accuracy):
        T.extend(Tr_P1)
    else:
        if((Tree[3]!=[0])&(Tree[3]!=[1])):
            Tr_P1=Prune_Tree(Tree[3],X_val,y_val)
    return T



Tp=Prune_Tree(Tree,X_trainv,y_trainv)


# Part d.

X_trains=X_train[0:490]
y_trains=y_train[0:490]
Accuracy=Testing(X_trains,y_trains,Tp)
print('Training Accuracy:',Accuracy)

X_trainv=X_train[490:560]
y_trainv=y_train[490:560]
Accuracy=Testing(X_trainv,y_trainv,Tp)
print('Validation Accuracy:',Accuracy)

X_test=X_train[560:699]
y_test=y_train[560:699]
Accuracy=Testing(X_test,y_test,Tp)
print('Test Accuracy:',Accuracy)

