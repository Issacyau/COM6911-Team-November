# Imports
import preprocessing
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
import seaborn as sns
import itertools

class LSTM_CNN(nn.Module):
    def __init__(self, input_size, width, output_size, lstm_size = 128, kern_1 = 5, kern_2 = 3):
        super().__init__()
        self.input_size  = input_size
        self.width = width
        self.output_size = output_size
        self.lstm_size = lstm_size
        self.kern_1 = kern_1
        self.kern_2 = kern_2
        
        # LAYERS
        self.lstm = nn.LSTM(self.input_size, self.lstm_size, num_layers = 2, batch_first=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size = self.kern_1, stride = (self.kern_1-1)//2)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = self.kern_2, stride = (self.kern_2-1)//2)
        self.globalavgpool = nn.AvgPool2d(kernel_size = (self.width//2 - 6, self.input_size//2 - 6), stride=1)
        self.batchnorm = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, self.output_size)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x):
        x = torch.squeeze(x)
        self.batch_size, self.height, self.width = x.shape
        x = x.reshape(self.batch_size, self.width, self.height)
        x, _ = self.lstm(x)
        x = x.reshape(self.batch_size, 1, self.width, self.lstm_size)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        self.globalavgpool.kernel_size = x.shape[-2:]
        x = self.globalavgpool(x)
        x = torch.squeeze(x)     
        x = self.batchnorm(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

if os.path.exists('results.pkl'):
    with open('results.pkl', 'rb') as results_file:
        results = pickle.load(results_file)
    print(results)
else:
    results = pd.DataFrame(columns = ['dataset', 'window_size', 'kernel_size', 'lstm_size', 'train_accuracy', 'test_accuracy'])
datasets = ['UCI', 'UTD', 'USC']

# Training parameters for each dataset
UCI_train_parameters = {'batch_size': 200, 'learning_rate': 0.005, 'num_epochs': 30}
UTD_train_parameters = {'batch_size': 25, 'learning_rate': 0.005, 'num_epochs': 100}
USC_train_parameters = {'batch_size': 200, 'learning_rate': 0.005, 'num_epochs': 15}

training = {'UCI': UCI_train_parameters, 'UTD': UTD_train_parameters, 'USC': USC_train_parameters}

# Class labels for each set
classes = {
'UCI': ['Walking', 'Walking upstairs', 'Walking downstairs', 'Sitting', 'Standing', 'Laying'], 
'UTD': ['right arm swipe to the left', 'right arm swipe to the right', 'right hand wave', 'two hand front clap',
        'right arm throw', 'cross arms in the chest', 'basketball shoot', 'right hand draw x', 
        'right hand draw circle (clockwise)', 'right hand draw circle (counter clockwise)',  'draw triangle', 'bowling (right hand)', 
        'front boxing', 'baseball swing from right', 'tennis right hand forehand swing', 'arm curl (two arms)', 'tennis serve',
        'two hand push', 'right hand knock on door', 'right hand catch an object', 'right hand pick up and throw'],
'USC': ['walking forward', 'walking left', 'walking right', 'walking upstairs', 'walking downstairs', 'running forward','jumping',
        'sitting', 'standing', 'sleeping', 'elevator up', 'elevator down']
}

# Hyper-parameters
window_sizes = [128, 64, 32]
lstm_sizes = [64, 32]
kernel_sizes = [7, 5, 3, (5, 3)]

for dataset, window_size in itertools.product(datasets, window_sizes):
    train_parameters = training[dataset]
    batch_size = train_parameters['batch_size']
    learning_rate = train_parameters['learning_rate']
    num_epochs = train_parameters['num_epochs']

    train_features = np.load(f'../Data/Processed/{dataset}/{dataset}_train_features_raw_{window_size}.npy', allow_pickle=True)
    train_labels = np.load(f'../Data/Processed/{dataset}/{dataset}_train_labels_{window_size}.npy', allow_pickle=True)
    test_features = np.load(f'../Data/Processed/{dataset}/{dataset}_test_features_raw_{window_size}.npy', allow_pickle=True)
    test_labels = np.load(f'../Data/Processed/{dataset}/{dataset}_test_labels_{window_size}.npy', allow_pickle=True)

    # Subtract 1 for every label for correct training, i.e., [1,2,3,4,5,6] to [0,1,2,3,4,5]
    train_labels = train_labels.astype(int) - 1
    test_labels = test_labels.astype(int) - 1

    # Transform to torch tensor
    tensor_train_features = torch.Tensor(train_features)
    tensor_train_labels = torch.Tensor(train_labels)
    tensor_test_features = torch.Tensor(test_features)
    tensor_test_labels = torch.Tensor(test_labels)

    # Add one dimension of channel
    tensor_train_features = torch.unsqueeze(tensor_train_features, 1)
    tensor_test_features = torch.unsqueeze(tensor_test_features, 1)

    # Create datset
    train_dataset = TensorDataset(tensor_train_features, tensor_train_labels)
    test_dataset = TensorDataset(tensor_test_features, tensor_test_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    height = train_features.shape[1]
    width = train_features.shape[2]
    output = len(classes[dataset])
    
    for lstm_size, kernel_size in itertools.product(lstm_sizes, kernel_sizes):
        if type(kernel_size) == int:
            kern_1, kern_2 = kernel_size, kernel_size
        else:
            kern_1, kern_2 = kernel_size

        comb_results = results.loc[(results['dataset'] == dataset) & \
            (results['window_size'] == window_size) & \
            (results['lstm_size'] == lstm_size) & \
            (results['kernel_size'] == kernel_size)]

        comb_done = len(comb_results) == 1

        if comb_done:
            continue
        else:
            print(f'Training with: {dataset} dataset, window size {window_size}, lstm size {lstm_size} and kernel size(s) {kernel_size}:')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            model = LSTM_CNN(height, width, output, lstm_size=lstm_size, kern_1=kern_1, kern_2=kern_2).to(device) #GPU

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            total_step = len(train_loader)
            loss_list = []
            acc_list = []
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):
                    # Run the forward pass
                    outputs = model(images.to(device))
                    loss = criterion(outputs, labels.to(device).long())
                    loss_list.append(loss.item())

                    # Backprop and perform Adam optimisation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track the accuracy
                    total = labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels.to(device).long()).sum().item()
                    acc_list.append(correct / total)
                print(f'Epoch: {epoch+1}/{num_epochs}')
                print(f'Accuracy: {correct/total}')
            
            model.eval()
            with torch.no_grad():
                
                correct = 0
                total = 0
                for images, labels in train_loader:
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    batch_total = labels.size(0)
                    batch_correct = (predicted == labels.to(device).long()).sum().item()
                    total += batch_total
                    correct += batch_correct

                train_accuracy = correct/total

                correct = 0
                total = 0
                for images, labels in test_loader:
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    batch_total = labels.size(0)
                    batch_correct = (predicted == labels.to(device).long()).sum().item()
                    total += batch_total
                    correct += batch_correct
            
                test_accuracy = correct/total
            model.train()

            new_entry = {'dataset': dataset, 'window_size': window_size, 'kernel_size': kernel_size, 'lstm_size': lstm_size, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
            results = results.append(new_entry, ignore_index = True)
            print(f'Result added: {new_entry}')
            with open('results.pkl', 'wb') as results_file:
                pickle.dump(results, results_file)
            results.to_csv('results_2.csv', index = False)