import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from rnn_model import RNNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train = pd.read_csv('cleaned_datasets/train/I04_train.csv', dtype=np.int64, index_col=0)
test = pd.read_csv('cleaned_datasets/test/I04_test.csv', dtype=np.int64, index_col=0)

train_np_target = train.loc[:, train.columns == 'result'].values
train_np_features = train.loc[:, train.columns != 'result'].values

test_np_target = test['result'].values
test_np_features = test.loc[:, test.columns != 'result'].values


train_target_tensor = torch.from_numpy(train_np_target).type(torch.FloatTensor)
train_features_tensor = torch.from_numpy(train_np_features).type(torch.FloatTensor)

test_target_tensor = torch.from_numpy(test_np_target).type(torch.FloatTensor)
test_features_tensor = torch.from_numpy(test_np_features).type(torch.FloatTensor)

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(train_np_features) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train_ds = TensorDataset(train_features_tensor,train_target_tensor)
test_ds = TensorDataset(test_features_tensor,test_target_tensor)

# data loader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Create RNN
input_dim = 66    # input dimension
hidden_dim = 100  # hidden layer dimension
layer_dim = 1     # number of hidden layers
output_dim = 1   # output dimension

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).cuda()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader): 
        features = features.to(device)
        labels = labels.to(device)

        features = features.reshape(features.shape[0], 1,-1)

        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(features)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 250 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for features, labels in test_loader:
                labels = labels.to(device)
                features = features.reshape(features.shape[0], 1,-1).to(device)

                # Forward propagation
                outputs = model(features)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            
            if count % 500 == 0:
                # Print Loss
                print(f'Iteration: {count}  Loss: {loss.data} Accuracy: {accuracy} %')
