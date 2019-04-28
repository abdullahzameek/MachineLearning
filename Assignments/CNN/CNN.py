import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## follow the instructions on the website
def unpickle(file):
    ## used to read binary files since our data files are in binary format
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

## loading data from binary data files
batch_1_dictionary = unpickle('cifar-10-data/data_batch_1')
batch_2_dictionary = unpickle('cifar-10-data/data_batch_2')

batch_1_dictionary.keys()

## get training, validation and testing sets
X_train_all = np.array(batch_1_dictionary[b'data']).reshape(10000,3,32,32)
y_train_all = np.array(batch_1_dictionary[b'labels'])
validation_count = 1000
train_count = X_train_all.shape[0]-1000
X_train = X_train_all[:train_count]
y_train = y_train_all[:train_count]
X_val = X_train_all[train_count:]
y_val = y_train_all[train_count:]
X_test = np.array(batch_2_dictionary[b'data']).reshape(10000,3,32,32)
y_test = np.array(batch_2_dictionary[b'labels'])

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

# for RGB data we can simply divide by 255
X_train_normalized = X_train/255
X_val_normalized =  X_val/255
X_test_normalized = X_test/255


## class label related
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def index_to_class_name(y):
    return CLASSES[y]
def class_name_to_index(class_name):
    return CLASSES.index(class_name)


import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
"""
Plotting utilities, if you want to know how these work exactly, check the reference
Or the documentations
reference: 
https://matplotlib.org/users/image_tutorial.html
https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
""" 

def show_single_image(data):
    # show a single image
    ## note that using matplotlib plotting function, we will have to reshape the data as (1,32,32,3)
    img = data.reshape(3,32,32).transpose(1,2,0)
    imgplot = plt.imshow(img)
def show_multiple_images(data, data_y, n_show=12, columns=4):
    ## given an array of data, show all of them as images
    fig=plt.figure(figsize=(8, 8))
    n = min(data.shape[0], n_show)
    rows = math.ceil(n/columns)
    for i in range(n):
        img = data[i].reshape(3,32,32).transpose(1,2,0)
        ax = fig.add_subplot(rows, columns, i+1) ## subplot index starts from 1 not 0
        class_name = index_to_class_name(data_y[i])
        ax.set_title(str(data_y[i])+": "+class_name)
        plt.imshow(img)
    plt.show()

show_multiple_images(X_train, y_train)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

class MLP(nn.Module):
    ## a very simple MLP model
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32,output_dim)

    def forward(self, x):
        z_1 = self.fc1(x)
        a_1 = F.relu(z_1)
        z_2 = self.fc2(a_1)
        return z_2
# utility for getting prediction accuracy
def get_correct_and_accuracy(y_pred, y):
    # y_pred is the nxC prediction scores
    # give the number of correct and the accuracy
    n = y.shape[0]
    # find the prediction class label
    _ ,pred_class = y_pred.max(dim=1)
    correct = (pred_class == y).sum().item()
    return correct ,correct/n


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# init network
conv_net = ConvNet()
print('model structure: ', conv_net)
optimizer = optim.Adam(conv_net.parameters(), lr= 1e-3) #lr = 1e-1 <==> learning rate = 0.001
# set loss function
criterion = nn.CrossEntropyLoss()

# prepare for mini-batch stochastic gradient descent
n_iteration = 40
batch_size = 256
n_train_data = X_train_normalized.shape[0]
n_val_data = X_val_normalized.shape[0]
n_test_data = X_test_normalized.shape[0]

n_batch = int(np.ceil(n_train_data/batch_size))


# convert X_train and X_val to tensor and flatten them
X_train_tensor = Tensor(X_train_normalized)
print(X_train_tensor.shape)

X_val_tensor = Tensor(X_val_normalized).reshape(n_val_data,3,32,32)
print(X_val_tensor.shape)

X_test_tensor = Tensor(X_test_normalized).reshape(n_test_data,-1)
print(X_test_tensor.shape)

# convert training label to tensor and to type long
y_train_tensor = Tensor(y_train).long()
y_val_tensor = Tensor(y_val).long()
y_test_tensor = Tensor(y_test).long()


print('X train tensor shape:', X_train_tensor.shape)

## start 
n_iteration = 40
train_loss_list = np.zeros(n_iteration)
train_accu_list = np.zeros(n_iteration)
val_loss_list = np.zeros(n_iteration)
val_accu_list = np.zeros(n_iteration)

train_accu = 0
val_accu = 0
ave_train_loss = 0
ave_val_loss = 0

for i in range(n_iteration):
    # first get a minibatch of data
    for j in range(n_batch):
        batch_start_index = j*batch_size
        # get data batch from the normalized data
        X_batch = X_train_tensor[batch_start_index:batch_start_index+batch_size]
        # get ground truth label y
        y_batch = y_train_tensor[batch_start_index:batch_start_index+batch_size]

        y_pred = conv_net.forward(X_batch)
        train_loss = criterion(y_pred, y_batch)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        numCorrect, iter_accu = get_correct_and_accuracy(y_pred,y_batch)
        
        train_accu += iter_accu
        ave_train_loss += train_loss
  
    train_accu /= n_batch
    ave_train_loss /= n_batch
    
    val_pred = conv_net.forward(X_val_tensor)
    val_loss = criterion(val_pred, y_val_tensor)
    optimizer.zero_grad()
    val_loss.backward()
    optimizer.step()
    correct, val_accu = get_correct_and_accuracy(val_pred,y_val_tensor)
    
    print("Iter %d ,Train loss: %.3f, Train acc: %.3f, Val loss: %.3f, Val acc: %.3f" 
          %(i ,ave_train_loss, train_accu, val_loss, val_accu)) 
    
    ## add to the logs so that we can use them later for plotting
    train_loss_list[i] = ave_train_loss
    train_accu_list[i] = train_accu
    val_loss_list[i] = val_loss
    val_accu_list[i] = val_accu
    
## plot training loss versus validation loss
x_axis = np.arange(n_iteration)
plt.plot(x_axis, train_loss_list, label='train loss')
plt.plot(x_axis, val_loss_list, label='val loss')
plt.legend()
plt.show()

## plot training accuracy versus validation accuracy
plt.plot(x_axis, train_accu_list, label='train acc')
plt.plot(x_axis, val_accu_list, label='val acc')
plt.legend()
plt.show()
