# # Q 1  [SVM Implementation]

# Data and Packages Loading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


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


# Q1 (a)

# Training for different Values of C

from sklearn import preprocessing
X_train=preprocessing.scale(X_train)
X_test=preprocessing.scale(X_test)
from sklearn.svm import LinearSVC
clf = LinearSVC(C=1e-4,random_state=0, tol=1e-5)
clf.fit(X_train[0:48000], y_train[0:48000])  


clf1 = LinearSVC(C=1e-3,random_state=0, tol=1e-5)
clf1.fit(X_train[0:48000], y_train[0:48000]) 


clf2 = LinearSVC(C=1e-2,random_state=0, tol=1e-5)
clf2.fit(X_train[0:48000], y_train[0:48000])


clf3 = LinearSVC(C=1e-1,random_state=0, tol=1e-5)
clf3.fit(X_train[0:48000], y_train[0:48000])


clf4 = LinearSVC(C=1,random_state=0, tol=1e-5)
clf4.fit(X_train[0:48000], y_train[0:48000])


clf5 = LinearSVC(C=1e1,random_state=0, tol=1e-5)
clf5.fit(X_train[1:48000], y_train[1:48000])


clf6 = LinearSVC(C=1e2,random_state=0, tol=1e-5)
clf6.fit(X_train[0:48000], y_train[0:48000])


clf7 = LinearSVC(C=1e3,random_state=0, tol=1e-5)
clf7.fit(X_train[0:48000], y_train[0:48000])


clf8 = LinearSVC(C=1e4,random_state=0, tol=1e-5)
clf8.fit(X_train[0:48000], y_train[0:48000])


# Computation of Testing, Validation and Training Accuracies

CL=[clf,clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8]
C_mat=np.array([1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4])
Test_Accuracy_test=np.zeros(len(C_mat))
Test_Accuracy_valid=np.zeros(len(C_mat))
Test_Accuracy_Train=np.zeros(len(C_mat))
X_valid=X_train[48000:60000]
y_valid=y_train[48000:60000]
X_T=X_train[0:48000]
y_T=y_train[0:48000]
for i in range(C_mat.shape[0]):
    Mistake=0
    for k in range(X_test.shape[0]):
        yp=CL[i].predict([X_test[k]])
        if(yp!=y_test[k]):
            Mistake=Mistake+1
    Test_Accuracy_test[i]=(len(X_test)-Mistake)/len(X_test)
    Mistake=0
    for k in range(X_T.shape[0]):
        yp=CL[i].predict([X_train[k]])
        if(yp!=y_T[k]):
            Mistake=Mistake+1
    Test_Accuracy_Train[i]=(len(X_train)-Mistake)/len(X_train)
    Mistake=0
    for k in range(X_valid.shape[0]):
        yp=CL[i].predict([X_valid[k]])
        if(yp!=y_valid[k]):
            Mistake=Mistake+1
    Test_Accuracy_valid[i]=(len(X_valid)-Mistake)/len(X_valid)


plt.plot(C_mat,Test_Accuracy_valid,label="Validation")
plt.plot(C_mat,Test_Accuracy_test,label="Test")
plt.plot(C_mat,Test_Accuracy_Train,label="Train")
plt.xscale('log',basex=10) 
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves SVM')
plt.legend()
plt.show()

# Q1 (b)

clfbest = LinearSVC(C=1e-1,random_state=0, tol=1e-5)
clfbest.fit(X_train, y_train)

X_train.shape


from sklearn.metrics import confusion_matrix
Mistake=0;
yp=np.zeros(X_test.shape[0])
for k in range(X_test.shape[0]):
        yp[k]=clfbest.predict([X_test[k]])
        if(yp[k]!=y_test[k]):
            Mistake=Mistake+1
Test_Accuracy_best=(len(X_test)-Mistake)/len(X_test)
print(Test_Accuracy_best)
## CONFUSION MATRIX
Conf_Mat=confusion_matrix(y_test, yp)
import pandas as pd
Conf_Table=(pd.DataFrame(Conf_Mat,columns=["Tshirt/Top", "Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle-Boot"],
                         index=["Tshirt/Top", "Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle-Boot"]))
print(Conf_Table)


# Q1 (c)

# Training for different degrees corresponding to best C value (1e-1).

clfpol1 = svm.SVC(C=1e-1,kernel='poly',degree=2)
clfpol1.fit(X_train[0:48000], y_train[0:48000])


clfpol2 = svm.SVC(C=1e-1,kernel='poly',degree=3)
clfpol2.fit(X_train[0:48000], y_train[0:48000])


clfpol3 = svm.SVC(C=1e-1,kernel='poly',degree=4)
clfpol3.fit(X_train[0:48000], y_train[0:48000])


CL=[clfpol1,clfpol2,clfpol3]
Degrees=np.array([2,3,4])
Test_Accuracy_test_poly=np.zeros(len(Degrees))
Test_Accuracy_valid_poly=np.zeros(len(Degrees))
Test_Accuracy_Train_poly=np.zeros(len(Degrees))
X_valid=X_train[48000:60000]
y_valid=y_train[48000:60000]
X_T=X_train[0:48000]
y_T=y_train[0:48000]
for i in range(Degrees.shape[0]):
    Mistake=0
    for k in range(X_test.shape[0]):
        yp=CL[i].predict([X_test[k]])
        print(k)
        if(yp!=y_test[k]):
            Mistake=Mistake+1
    Test_Accuracy_test_poly[i]=(len(X_test)-Mistake)/len(X_test)
    print("Done1")
    Mistake=0
    for k in range(X_valid.shape[0]):
        print(k)
        yp=CL[i].predict([X_valid[k]])
        if(yp!=y_valid[k]):
            Mistake=Mistake+1
    Test_Accuracy_valid_poly[i]=(len(X_valid)-Mistake)/len(X_valid)
    print('Done2')
    Mistake=0
    for k in range(X_T.shape[0]):
        print(k)
        yp=CL[i].predict([X_train[k]])
        if(yp!=y_T[k]):
            Mistake=Mistake+1
    Test_Accuracy_Train_poly[i]=(len(X_train)-Mistake)/len(X_train)
    print('Done3')


print(Test_Accuracy_valid_poly)
print(Test_Accuracy_Train_poly)
print(Test_Accuracy_test_poly)
Degre=np.array([2,3,4])
plt.plot(Degre,Test_Accuracy_valid_poly,label="Validation")
plt.plot(Degre,Test_Accuracy_test_poly,label="Test")
plt.plot(Degre,Test_Accuracy_Train_poly,label="Train")
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves SVM- Polynomial Kernel')
plt.legend()
plt.show()


NSV=np.zeros(3)
NSV[0]=len(CL[0].support_vectors_)
NSV[1]=len(CL[1].support_vectors_)
NSV[2]=len(CL[2].support_vectors_)
plt.plot(Degre,NSV)
plt.xlabel('Degree')
plt.ylabel('# of SVs')
plt.title('Number of Support Vectors- Polynomial Kernel')
plt.show()

