import os
import h5py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from architectures import *

#constants
model_path = "/media/data/jsettlers/machine_learning/pretrained_basic.pt"

#hyper-parameters
batch_size     = 100
learning_rate  = 5*1e-4

#READ DATA
#option 1
#with h5py.File("./DATASET/dataset.h5", "r") as file:
#    inputs_h5  = file["inputs"][:]
#    outputs_h5 = file["outputs"][:]

#option 2
inputs_npy  = np.load("./DATASET/Preprocessed_Inputs.npy")
outputs_npy = np.load("./DATASET/Preprocessed_Outputs.npy")

NN_inputs  = torch.tensor(inputs_npy, dtype=torch.float)
NN_outputs = torch.tensor(outputs_npy, dtype=torch.float)
#NN_outputs = torch.max(torch.tensor(outputs_npy, dtype=torch.int), 1)[1]
print(f'Input shape:  {NN_inputs.shape}')
print(f'Output shape: {NN_outputs.shape}')
print('='*100)
#print(NN_inputs[0])
#print(NN_outputs[0])

#CREATE TRAINING, VALIDATION & TEST SETS
inputs_train, inputs_rem, outputs_train, outputs_rem = train_test_split(NN_inputs, NN_outputs, test_size=0.2)   #random_state parameter can create re-producible results.
intputs_val, inputs_test, outputs_val, outputs_test  = train_test_split(inputs_rem, outputs_rem, test_size=0.5) #random_state parameter can create re-producible results.

trainset = TensorDataset(inputs_train, outputs_train)
valset   = TensorDataset(intputs_val, outputs_val) 
testset  = TensorDataset(inputs_test, outputs_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False)

#CREATE MODEL, CRITERION & OPTIMIZER
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = basic().to(device)
if os.path.exists(model_path):
    print('Model Already Exists!\nWe\'re loading it...')
    model.load_state_dict(torch.load(model_path))
    print('='*100)

criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters())
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

print("Device:", device)
print(model)

#TRAIN MODEL
epochs = 10
v_epochs     = [i for i in range(1, epochs+1)]
v_train_loss = [0 for _ in range(epochs)]
v_train_acc  = [0 for _ in range(epochs)]
v_val_loss   = [0 for _ in range(epochs)]
v_val_acc    = [0 for _ in range(epochs)]

start_time = time.time()
for epoch in range(epochs):
    model.train()
    print(f'Epoch {epoch+1}:')
    for i, data in enumerate(trainloader, 0):
        batch_inputs, batch_labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        #if i % 1000 == 999:    # print every 1000 mini-batches
        #    print(f'Iteration {i+1} Training Loss: {(running_loss / 1000):.3f}')
        #    running_loss = 0.0

    model.eval()
    with torch.no_grad():
        total = 0
        counter = 0
        correct = 0
        running_loss = 0
        for data in trainloader:
            batch_inputs, batch_labels = data[0].to(device), data[1].to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
    
            _, predictions = torch.max(outputs, dim=1)
            _, actual      = torch.max(batch_labels, dim=1)
            correct += (predictions == actual).sum().item()
            total += batch_size
            running_loss += loss.item()
            counter += 1 
        print(f"Training Loss: {(running_loss / counter):.3f}")
        print(f"Training Accuracy: {(100 * correct / total):.0f}%")
        v_train_loss[epoch] = running_loss / counter
        v_train_acc[epoch]  = 100 * correct / total

    model.eval()
    with torch.no_grad():
        total = 0
        counter = 0
        correct = 0
        running_loss = 0
        for data in validloader:
            val_inputs, val_labels = data[0].to(device), data[1].to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
    
            _, actual      = torch.max(val_labels, dim=1)
            _, predictions = torch.max(val_outputs, dim=1)
            correct += (predictions == actual).sum().item()
            total += batch_size
            running_loss += val_loss.item()
            counter += 1
        print(f'Validation Loss: {(running_loss/counter):.3f}')
        print(f'Validation Accuracy: {(100 * correct / total):.0f}%')
        print(f'Epoch [{epoch+1}/{epochs}] completed!\n')
        v_val_loss[epoch] = running_loss / counter
        v_val_acc[epoch]  = 100 * correct / total
end_time = time.time()
training_time = end_time - start_time
print('Finished Training')
print(f"Training time: {training_time:.2f} seconds")

#EVALUATE MODEL USING TEST SET
model.eval()
with torch.no_grad():
    total = 0
    counter = 0
    correct = 0
    running_loss = 0
    for data in testloader:
        test_inputs, test_labels = data[0].to(device), data[1].to(device)
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_labels)
        
        _, actual      = torch.max(test_labels, dim=1)
        _, predictions = torch.max(test_outputs, 1)
        correct += (predictions == actual).sum().item()
        total += batch_size
        running_loss += test_loss.item()
        counter += 1
        
print(f'\nTest Loss: {(running_loss/counter):.3f}')
print(f'Accuracy of the network on {total} test games: {(100 * correct / total):.0f}%\n')

print(f'Saving model at: {model_path}\n')
torch.save(model.state_dict(), model_path)

#PLOT RESULTS
plt.figure()
plt.plot(v_epochs, v_train_loss, label='train')
plt.plot(v_epochs, v_val_loss, label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
#plt.title('BASIC NETWORK | 1 DROID, 3 RANDOM')
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(v_epochs, v_train_acc, label='train')
plt.plot(v_epochs, v_val_acc, label='validation')
plt.xlabel('epochs')
plt.ylabel('accuracy')
#plt.title('BASIC NETWORK | 1 DROID, 3 RANDOM')
plt.legend()
plt.grid(True)

#plt.show()
