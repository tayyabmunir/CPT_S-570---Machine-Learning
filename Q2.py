#!/usr/bin/env python
# coding: utf-8

# # Deep Learning

# In[2]:


import matplotlib.pyplot as plt
from torch import tensor
import torch
import matplotlib as mpl
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# ## Loading dataset

# In[3]:


def get_data():
    train_data = pd.read_csv('./data/fashion-mnist_train.csv')
    test_data = pd.read_csv('./data/fashion-mnist_test.csv')
    x_train = train_data[train_data.columns[1:]].values
    y_train = train_data.label.values
    x_test = test_data[test_data.columns[1:]].values
    y_test = test_data.label.values
    return map(tensor, (x_train, y_train, x_test, y_test)) # maps are useful functions to know
                                                           # here, we are just converting lists to pytorch tensors


# In[5]:


x_train, y_train, x_test, y_test = get_data()
train_n, train_m = x_train.shape
test_n, test_m = x_test.shape
n_cls = y_train.max()+1


# In[24]:


y_train=y_train[0:10]
x_train=x_train[0:10]


# In[6]:


train_n, train_m = x_train.shape
test_n, test_m = x_test.shape
n_cls = y_train.max()+1


# ## Creating a model
# Convolutional Neural Network Model is created in this part:
# * First Convolutional Layer has 8 Outputs, stride length of 2, and Filter size of 5x5 is used. This produces 8 outputs of 14x14 convolutional layers. Convolutional layers are followed by RELU layer.
# * Second covolutional layer has 16 outputs each a 7x7 convolutional layer. Convolutional layers are followed by RELU layer.
# * Third Convolutional layer: This produces 32 outputs each a 4x4 convolutional layer. Convolutional layers are followed by RELU layer.
# * Fourth Convolutional layer: This produces 32 outputs each a 2x2 convolutional layer. Convolutional layers are followed by RELU layer.
# * Polling Layer: Polling layer produces 32 outputs each is 1x1 scalar.
# * Fully connected layer takes 32 1x1 inputs and 10 class labels
# * A relu activation function is also used after pooling layer to enhance accuracy.

# In[39]:


# Definition of the model
class FashionMnistNet(nn.Module):
    # Based on Lecunn's Lenet architecture
    def __init__(self):
        super(FashionMnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5,2,2) 
        self.conv2 = nn.Conv2d(8, 16, 3,2,1)
        self.conv3 = nn.Conv2d(16, 32, 3,2,1)
        self.conv4 = nn.Conv2d(32, 32, 3,2,1)
        self.mp=nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = F.relu(self.conv3(x))
        #print(x.size())
        x=self.conv4(x)
        #print(x.size())
        x = F.relu(self.mp(x))
        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())
        x = self.fc(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# ## Training the model

# In[40]:


model = FashionMnistNet()
print(model)
model.forward(x_train[0].reshape(1, 1, 28, 28))


# In[31]:


### Normalization
x_train, x_test = x_train.float(), x_test.float()
train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std


# In[32]:


def normalize(x, m, s): return (x-m)/s
x_train = normalize(x_train, train_mean, train_std)
x_test = normalize(x_test, train_mean, train_std) # note this normalize test data also with training mean and standard deviation


# In[34]:


model_wnd = FashionMnistNet()
lr = 0.1 # learning rate
epochs = 10 # number of epochs
bs = 32
loss_func = F.cross_entropy
opt = optim.ASGD(model_wnd.parameters(), lr=lr)
accuracy_vals_wnd = []
train_accuracy_vals_wnd = []
for epoch in range(epochs):
    model_wnd.train()
    for i in range((train_n-1)//bs + 1):
        random_idxs = torch.randperm(train_n) 
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i].reshape(bs, 1, 28, 28)
        yb = y_train[start_i:end_i]
        loss = loss_func(model_wnd.forward(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    model_wnd.eval()
    with torch.no_grad():
        total_loss, accuracy = 0., 0.
        validation_size = int(test_n/10)
        for i in range(test_n):
            x = x_test[i].reshape(1, 1, 28, 28)
            y = y_test[i]
            pred = model_wnd.forward(x)
            accuracy += (torch.argmax(pred) == y).float()
        print("Accuracy_Test: ", (accuracy*100/test_n).item(),test_n,accuracy)
        accuracy_vals_wnd.append((accuracy*100/test_n).item())
    model_wnd.eval()
    with torch.no_grad():
        total_loss_train, accuracy_train = 0., 0.
        validation_size = int(train_n/10)
        for i in range(train_n):
            x = x_train[i].reshape(1, 1, 28, 28)
            y = y_train[i]
            pred = model_wnd.forward(x)
            accuracy_train += (torch.argmax(pred) == y).float()
        print("Accuracy_Train: ", (accuracy_train*100/train_n).item(),test_n,accuracy_train)
        train_accuracy_vals_wnd.append((accuracy_train*100/train_n).item())


# In[38]:


P1=plt.plot(accuracy_vals_wnd)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
P2=plt.plot(train_accuracy_vals_wnd)
plt.legend(('Test-Accuracy','Train-Accuracy'))

