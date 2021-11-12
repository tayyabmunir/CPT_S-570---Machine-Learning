import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Import and Check
Data=pd.read_csv('C:/Users/tayya/OneDrive/Desktop/Fall 2020/Machine Learning/Assignment 1/A1/fashion-mnist_train.csv')
Data_test=pd.read_csv('C:/Users/tayya/OneDrive/Desktop/Fall 2020/Machine Learning/Assignment 1/A1/fashion-mnist_test.csv')


X_train=np.zeros([len(Data),784])
y_train=np.zeros(len(Data))
X_test=np.zeros([len(Data_test),784])
y_test=np.zeros(len(Data_test))

for i in range(len(Data)):
    y_train[i]=Data.loc[i,'label']
    if y_train[i]%2==0:
        y_train[i]=1
    else:
        y_train[i]=-1
    X_train[i]=Data.loc[i,'pixel1':'pixel784']

for i in range(len(Data_test)):
    y_test[i]=Data_test.loc[i,'label']
    if y_test[i]%2==0:
        y_test[i]=1
    else:
        y_test[i]=-1
    X_test[i]=Data_test.loc[i,'pixel1':'pixel784']

def Ch(W,X_train,y_new):
    Train_Rights=0;
    Train_Errors=0;
    for p in range(len(X_train)):
        yh=np.sign(np.matmul(W,X_train[p]));
        if yh==y_new[p]:
            Train_Rights=Train_Rights+1;
        elif yh!=y_new[p]:
            Train_Errors=Train_Errors+1;
    return Train_Errors

def Train_Perceptron(X_train,y_train,T):
    tau=1;
    Errors=np.array([])
    Wa=np.zeros([T,len(X_train[0])])
    W=np.zeros(len(X_train[0]))
    for j in range(T):
        Mis=0
        for p in range(len(X_train)):
            yh=np.sign(np.matmul(W,X_train[p]));
            if np.sign(yh)==np.sign(y_train[p]):
                W=W
            elif np.sign(yh)!=np.sign(y_train[p]):
                W=W+tau*y_train[p]*X_train[p]
                Mis=Mis+1;
        Errors=np.append(Errors,Mis)
        Wa[j]=W;
    return W,Errors,Wa

def Train_PassiveAggressive(X_train,y_train,T):
    tau=0;
    Errors=np.array([]);
    W=np.zeros(len(X_train[0]))
    Wa=np.zeros([T,len(X_train[0])])
    for j in range(T):
        Mis=0;
        for p in range(len(X_train)):
            yh=np.sign(np.dot(W,X_train[p]));
            if np.sign(yh)==np.sign(y_train[p]):
                W=W
            elif np.sign(yh)!=np.sign(y_train[p]):
                tau=(1-y_train[p]*np.dot(W,X_train[p]))/(np.linalg.norm(X_train[p]))**2
                W=W+tau*y_train[p]*X_train[p]
                Mis=Mis+1;
        Errors=np.append(Errors,Mis)
        Wa[j]=W;
    return W,Errors,Wa

def Train_Averaged_Perceptron(X_train,y_train,T):
    tau=1;
    Count=1
    Wsum=np.zeros(len(X_train[0]))
    Errors=np.array([])
    Wa=np.zeros([T,len(X_train[0])])
    W=np.zeros(len(X_train[0]))
    for j in range(T):
        Mis=0
        for p in range(len(X_train)):
            yh=np.sign(np.matmul(W,X_train[p]));
            if np.sign(yh)!=np.sign(y_train[p]):
                W=W+tau*y_train[p]*X_train[p]
                Wsum=Wsum+tau*y_train[p]*X_train[p]*Count
                Mis=Mis+1;
            Count=Count+1;
        Errors=np.append(Errors,Mis)
        Wa[j]=W-Wsum/Count;
    W=W-Wsum/Count
    return W,Errors,Wa

[W,Errors_Perceptron,Wa]=Train_Perceptron(X_train,y_train,50)
[W,Errors_PassiveAggressive,Wa]=Train_PassiveAggressive(X_train,y_train,50)

Errors_Perceptron

# a

X=range(1,51)
plt.plot(X,Errors_Perceptron)
plt.xlabel('Training Iterations')
plt.ylabel('Training Errors')
plt.title('Online Learning Curve for Perceptron-Algorithm')
plt.savefig('Part(a)Perceptron_Learning_Curve.png')
plt.show()

plt.plot(X,Errors_PassiveAggressive)
plt.xlabel('Training Iterations')
plt.ylabel('Training Errors')
plt.title('Online Learning Curve for Passive-Aggressive Algorithm')
plt.savefig('Part(a)PerceptronPassiveAggressive_Learning_Curve.png')
plt.show()

# b

[W,Errors_Perceptron,Wa_perc]=Train_Perceptron(X_train,y_train,20)
[W,Errors_PassiveAggressive,Wa_pa]=Train_PassiveAggressive(X_train,y_train,20)

Train_Accuracy_perceptron=np.zeros([Wa_pa.shape[0]])
Test_Accuracy_perceptron=np.zeros([Wa_pa.shape[0]])
Train_Accuracy_PassiveAggressive=np.zeros([Wa_pa.shape[0]])
Test_Accuracy_PassiveAggressive=np.zeros([Wa_pa.shape[0]])
for i in range(Wa_pa.shape[0]):
    Train_Accuracy_perceptron[i]=(len(X_train)-Ch(Wa_perc[i],X_train,y_train))/len(X_train)
    Test_Accuracy_perceptron[i]=(len(X_test)-Ch(Wa_perc[i],X_test,y_test))/len(X_test)
    Train_Accuracy_PassiveAggressive[i]=(len(X_train)-Ch(Wa_pa[i],X_train,y_train))/len(X_train)
    Test_Accuracy_PassiveAggressive[i]=(len(X_test)-Ch(Wa_pa[i],X_test,y_test))/len(X_test)
    

X=range(Wa_pa.shape[0])
plt.plot(X,Train_Accuracy_perceptron)
plt.title('Perceptron Train Accuracy Graph')
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Part(b)Perceptron Train Accuracy.png')
plt.show()

plt.plot(X,Test_Accuracy_perceptron, label='Percetron Test Accuracy')
plt.title('Percetron Test Accuracy Graph')
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Part(b)Percetron Test Accuracy.png')
plt.show()

plt.plot(X,Train_Accuracy_PassiveAggressive,label='Passive Aggressive Train Accuracy')
plt.title('Passive Aggressive Train Accuracy Graph')
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Part(b)Passive Aggressive Train Accuracy.png')
plt.show()


plt.plot(X,Test_Accuracy_PassiveAggressive,label='Passive Aggressive Test Accuracy')
plt.title('Passive Aggressive Test Accuracy Graph')
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Part(b)Passive Aggressive Test Accuracy.png')
plt.show()

# (c)

[W,Errors_averaged_Perceptron,Wa_averaged_perceptron]=Train_Averaged_Perceptron(X_train,y_train,20)


Test_Accuracy_averaged_perceptron=np.zeros([Wa_averaged_perceptron.shape[0]])
for i in range(Wa_averaged_perceptron.shape[0]):
    Test_Accuracy_perceptron[i]=(len(X_test)-Ch(Wa_perc[i],X_test,y_test))/len(X_test)
    Test_Accuracy_averaged_perceptron[i]=(len(X_test)-Ch(Wa_averaged_perceptron[i],X_test,y_test))/len(X_test)

X=range(Wa_averaged_perceptron.shape[0])
plt.plot(X,Test_Accuracy_perceptron,label= 'Perceptron')
plt.xlabel('Test Iterations')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Perceptron and Averaged Perceptron Algorithm Graph')
#plt.show()
plt.plot(X,Test_Accuracy_averaged_perceptron, label='Averaged Perceptron')
plt.xlabel('Test Iterations')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Perceptron and Averaged Perceptron Algorithm Graph')
plt.legend()
plt.savefig('Part(c)Test Accuracy of Perceptron and Averaged Perceptron Algorithm Graph.png')
plt.show()

# (d)

r=range(3000,61000,3000)
p=0
Accuracy_perceptron=np.zeros(len(r))
Accuracy_passive_aggressive=np.zeros(len(r))
Accuracy_percav=np.zeros(len(r))
for i in r:
    print(i)
    X_T=X_train[0:i]
    Y_T=y_train[0:i]
    [Wper,Errors_Perceptron,Wa_perc]=Train_Perceptron(X_T,Y_T,20)
    [Wpa,Errors_PassiveAggressive,Wa_pa]=Train_PassiveAggressive(X_T,Y_T,20)
    [Wperav,Errors_Perceptronav,Wa_percav]=Train_Averaged_Perceptron(X_T,Y_T,20)
    Accuracy_perceptron[p]=(len(X_test)-Ch(Wper,X_test,y_test))/len(X_test)
    Accuracy_passive_aggressive[p]=(len(X_test)-Ch(Wpa,X_test,y_test))/len(X_test)
    Accuracy_percav[p]=(len(X_test)-Ch(Wperav,X_test,y_test))/len(X_test)
    p+=1
    

X=r
plt.plot(X,Accuracy_perceptron,label='Perceptron')
plt.plot(X,Accuracy_passive_aggressive,label='Passive Aggressive')
plt.plot(X,Accuracy_percav,label='Average Perceptron')
plt.xlabel('Training Samples')
plt.ylabel('Test Accuracy')
plt.title('All Perceptron Algorithms Graph')
plt.legend()
plt.savefig('Part(d)All Perceptron Algorithm Graph.png')
plt.show()


