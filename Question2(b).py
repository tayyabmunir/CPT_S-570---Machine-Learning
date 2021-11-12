# Multi-Class Classification Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Data=pd.read_csv('C:/Users/tayya/OneDrive/Desktop/Fall 2020/Machine Learning/Assignment 1/A1/fashion-mnist_train.csv')
Data_test=pd.read_csv('C:/Users/tayya/OneDrive/Desktop/Fall 2020/Machine Learning/Assignment 1/A1/fashion-mnist_test.csv')

X_train=np.zeros([len(Data),784])
y_train=np.zeros(len(Data))
X_test=np.zeros([len(Data_test),784])
y_test=np.zeros(len(Data_test))


for i in range(len(Data)):
    y_train[i]=Data.loc[i,'label']
    X_train[i]=Data.loc[i,'pixel1':'pixel784']

for i in range(len(Data_test)):
    y_test[i]=Data_test.loc[i,'label']
    X_test[i]=Data_test.loc[i,'pixel1':'pixel784']

def Check(W,X_train,y_new,K):
    Train_Rights=0;
    Train_Errors=0;
    for p in range(len(X_train)):
        F=Form_Fashion(X_train[p],K)
        yh=np.argmax(np.matmul(F,W));
        if yh==y_new[p]:
            Train_Rights=Train_Rights+1;
        elif yh!=y_new[p]:
            Train_Errors=Train_Errors+1;
    return Train_Errors

def Train_Averaged_Perceptron(X_train,y_train,T,K):
    tau=1;
    Errors=np.array([])
    N=len(X_train[0])
    Wa=np.zeros([T,K*N,])
    W=np.zeros(K*N)
    Wsum=np.zeros(K*N)
    Count=1;
    for j in range(T):
        Mistakes=0
        for p in range(len(X_train)):
            F=Form_Fashion(X_train[p],K)
            yh=np.argmax(np.matmul(F,W));
            if yh!=y_train[p]:
                Wsum=Wsum+W*Count;
                W=W+tau*(F[int(y_train[p])]-F[int(yh)])
                Mistakes=Mistakes+1
                Count=1;
            Count=Count+1
        Errors=np.append(Errors,Mistakes)
        Wa[j]=Wsum
    W=Wsum
    print('Train_Averaged_Perceptron')
    return W,Errors,Wa

def Train_Passive_Aggressive(X_train,y_train,T,K):
    tau=np.ones(K);
    N=len(X_train[0])
    W=np.zeros(K*N)
    Wa=np.zeros([T,K*N,])
    Errors=np.array([])
    for j in range(T):
        Mistakes=0
        for p in range(len(X_train)):
            F=Form_Fashion(X_train[p],K)
            yh=np.argmax(np.matmul(F,W));
            if yh==y_train[p]:
                W=W
            elif yh!=y_train[p]:
                tau=Computation_tau(y_train[p],yh,W,F,K)
                W=W+tau*(F[int(y_train[p])]-F[int(yh)])
                Mistakes=Mistakes+1
        Errors=np.append(Errors,Mistakes)
        Wa[j]=W
    print('Complete_Train_Passive_Aggressive')
    return W,Errors,Wa

def Train_Perceptron(X_train,y_train,T,K):
    tau=np.ones(K);
    N=len(X_train[0])
    W=np.zeros(K*N)
    Wa=np.zeros([T,K*N,])
    Errors=np.array([])
    for j in range(T):
        Mistakes=0
        for p in range(len(X_train)):
            F=Form_Fashion(X_train[p],K)
            yh=np.argmax(np.matmul(F,W));
            if yh==y_train[p]:
                W=W
            elif yh!=y_train[p]:
                tau=1
                W=W+tau*(F[int(y_train[p])]-F[int(yh)])
                Mistakes=Mistakes+1
        Errors=np.append(Errors,Mistakes)
        Wa[j]=W
    print('Complete_Train_Perceptron')
    return W,Errors,Wa

def Form_Fashion(X_traini,K):
    N=len(X_traini)
    Fashion=np.zeros((K,K*N))
    for p in range(K):
        Fashion[p][p*N:(p+1)*N]=X_traini
    return Fashion

def Form_CF(X_traini,yt,K):
    N=len(X_traini)
    Fc=np.zeros(K*N)
    Fc[yt*N:(yt+1)*N]=X_traini
    return Fc

def Form_W(W,K):
    W_seg=np.zeros((K,784,))
    for i in range(K):
        W_seg[i]=W[i*784:(i+1)*784]
    return W_seg

def Computation_tau(yt,yh,W,F,K):
    K=10;
    tau=np.zeros(K);
    Diff=F[int(yt)]-F[int(yh)];
    tau=1-np.matmul(Diff,W)
    tau=tau/np.linalg.norm(Diff)**2
    return tau

# (a)

[W_Perceptron,Errors_Perceptron,Wa_Perceptron]=Train_Perceptron(X_train,y_train,50,10)
[W_Passive_Aggressive,Errors_Passive_Aggressive,Wa_Passive_Aggressive]=Train_Passive_Aggressive(X_train,y_train,50,10)

X=range(1,51)
plt.plot(X,Errors_Perceptron)
plt.xlabel('Training Iterations')
plt.ylabel('Training Errors')
plt.title('Online Learning Curve for Perceptron-Algorithm')
plt.savefig('Part2(a)Online Learning Curve for Perceptron-Algorithm.png')
plt.show()
plt.plot(X,Errors_Passive_Aggressive)
plt.xlabel('Training Iterations')
plt.ylabel('Training Errors')
plt.title('Online Learning Curve for Passive-Aggressive Algorithm')
plt.savefig('Part2(a)Online Learning Curve for Passive-Aggressive.png')
plt.show()


# (b)

[W_Perceptron,Errors_Perceptron,Wa_Perceptron]=Train_Perceptron(X_train,y_train,20,10)
[W_Passive_Aggressive,Errors_Passive_Aggressive,Wa_Passive_Aggressive]=Train_Passive_Aggressive(X_train,y_train,20,10)

Train_Accuracy_perceptron=np.zeros([Wa_Passive_Aggressive.shape[0]])
Test_Accuracy_perceptron=np.zeros([Wa_Passive_Aggressive.shape[0]])
Train_Accuracy_Passive_Aggressive=np.zeros([Wa_Passive_Aggressive.shape[0]])
Test_Accuracy_Passive_Aggressive=np.zeros([Wa_Passive_Aggressive.shape[0]])
for i in range(Wa_Passive_Aggressive.shape[0]):
    Train_Accuracy_perceptron[i]=(len(X_train)-Check(Wa_Perceptron[i],X_train,y_train,10))/len(X_train)
    Test_Accuracy_perceptron[i]=(len(X_test)-Check(Wa_Perceptron[i],X_test,y_test,10))/len(X_test)
    Train_Accuracy_Passive_Aggressive[i]=(len(X_train)-Check(Wa_Passive_Aggressive[i],X_train,y_train,10))/len(X_train)
    Test_Accuracy_Passive_Aggressive[i]=(len(X_test)-Check(Wa_Passive_Aggressive[i],X_test,y_test,10))/len(X_test)


X=range(Wa_Passive_Aggressive.shape[0])
plt.plot(X,Train_Accuracy_perceptron)
plt.xlabel('Training Iterations')
plt.ylabel('Training Accuracy')
plt.title('Graph for Perceptron-Algorithm Training')
plt.savefig('Part2(b) Perceptron Algorithm Graph')
plt.show()

plt.plot(X,Test_Accuracy_perceptron)
plt.xlabel('Training Iterations')
plt.ylabel('Test Accuracy')
plt.title('Graph for Perceptron-Algorithm Test and Training')
plt.savefig('Part2(b) Perceptron Algorithm Graph')
plt.show()

plt.plot(X,Train_Accuracy_Passive_Aggressive)
plt.xlabel('Training Iterations')
plt.ylabel('Training Accuracy')
plt.title('Graph for Passive Aggressive Algorithm Training')
plt.savefig('Part2(b) Perceptron Algorithm Graph')
plt.show()

plt.plot(X,Test_Accuracy_Passive_Aggressive)
plt.xlabel('Training Iterations')
plt.ylabel('Test Accuracy')
plt.title('Graph for Passive-Aggressive Algorithm Test and Training')
plt.savefig('Part2(b) Perceptron Algorithm Graph')
plt.show()

# (c)

[W_AVP,Errors_AVP,Wa_AVP]=Train_Averaged_Perceptron(X_train,y_train,20,10)

Train_Accuracy_avp_perceptron=np.zeros([Wa_AVP.shape[0]])
Test_Accuracy_avp_perceptron=np.zeros([Wa_AVP.shape[0]])
for i in range(Wa_AVP.shape[0]):
    Train_Accuracy_avp_perceptron[i]=(len(X_train)-Check(Wa_AVP[i],X_train,y_train,10))/len(X_train)
    Test_Accuracy_avp_perceptron[i]=(len(X_test)-Check(Wa_AVP[i],X_test,y_test,10))/len(X_test)

X=range(Wa_AVP.shape[0])

Test_Accuracy_avp_perceptron.shape
Train_Accuracy_avp_perceptron.shape
plt.plot(X,Train_Accuracy_avp_perceptron,label='Training Data')
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.plot(X,Test_Accuracy_avp_perceptron,label='Testing Data')
plt.legend()
plt.title('Graph for Averaged Perceptron-Algorithm Training and Testing')
plt.savefig('Part2(c)Graph for Averaged Perceptron-Algorithm Training and Testing ')
plt.show()

print(Train_Accuracy_avp_perceptron)

# (d)

k=range(3000,63000,3000)
p=0
Accuracy_perceptron=np.zeros(len(k))
Accuracy_perceptron_averaged=np.zeros(len(k))
Accuracy_passive_aggressive=np.zeros(len(k))
for i in k:
    print(i)
    X_T=X_train[0:i]
    Y_T=y_train[0:i]
    [Wper,Errors_Perceptron,Wa_perc]=Train_Perceptron(X_T,Y_T,20,10)
    [Wpa,Errors_PassiveAggressive,Wa_pa]=Train_Passive_Aggressive(X_T,Y_T,20,10)
    [Wperav,Errors_Perceptronav,Wa_percav]=Train_Averaged_Perceptron(X_T,Y_T,20,10)
    Accuracy_perceptron[p]=(len(X_test)-Check(Wpa,X_test,y_test,10))/len(X_test)
    Accuracy_passive_aggressive[p]=(len(X_test)-Check(Wper,X_test,y_test,10))/len(X_test)
    Accuracy_perceptron_averaged[p]=(len(X_test)-Check(Wperav,X_test,y_test,10))/len(X_test)
    p+=1 

X=k
plt.plot(X,Accuracy_perceptron,label='Perceptron')
plt.plot(X,Accuracy_perceptron_averaged,label='Average Perceptron')
plt.plot(X,Accuracy_passive_aggressive,label='Passive Aggressive')
plt.xlabel('Training Samples')
plt.ylabel('Testing Accuracy')
plt.title('Generalized Learning Curve for Perceptron, Averaged Perceptron and Passive Aggressive Algorithm')
plt.legend()
plt.savefig('Part2(d)Generalized Learning Curve for Perceptron, Averaged Perceptron and Passive Aggressive Algorithm')
plt.show()





