import os
import h5py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from architectures import *

#hyper-parameters
batch_size     = 100
learning_rate  = 5 * 1e-4

def train_model(model, optimizer, criterion, device, epochs, trainloader, validloader, testloader, result_queue):
    model.to(device)

    print(f'Device: {device}')
    print(model)
    print('='*100)

    v_train_loss = [0 for _ in range(epochs)]
    v_train_acc  = [0 for _ in range(epochs)]
    v_val_loss   = [0 for _ in range(epochs)]
    v_val_acc    = [0 for _ in range(epochs)]

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        #print(f'Network {model.__class__.__name__} | Epoch {epoch+1}:')
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
            #print(f"Training Loss: {(running_loss / counter):.3f}")
            #print(f"Training Accuracy: {(100 * correct / total):.0f}%")
            v_train_loss[epoch] = running_loss / counter
            v_train_acc[epoch]  = correct / total

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
            #print(f'Validation Loss: {(running_loss/counter):.3f}')
            #print(f'Validation Accuracy: {(100 * correct / total):.0f}%')
            print(f'Network {model.__class__.__name__} | Epoch [{epoch+1}/{epochs}] completed!')
            v_val_loss[epoch] = running_loss / counter
            v_val_acc[epoch]  = correct / total
    end_time = time.time()
    training_time = end_time - start_time

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
            
    #print(f'Network {model.__class__.__name__} | Finished Training')
    #print(f"Training time: {training_time:.2f} seconds")
    #print(f'Test Loss:     {(running_loss/counter):.3f}')
    #print(f'Accuracy of the network on {total} test games: {(100 * correct / total):.0f}%\n')
    
    result_queue.put(v_train_loss)
    result_queue.put(v_val_loss)
    result_queue.put(v_train_acc)
    result_queue.put(v_val_acc)
    
if __name__ == "__main__":
    #constants
    basic_model_path = "/media/data/dpapadam/jsettlers/machine_learning/pretrained_basic.pt"
    cnn_model_path  = "/media/data/dpapadam/jsettlers/machine_learning/pretrained_cnn.pt"

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
    dev0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dev1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model_basic = basic()
    model_cnn   = cnn()
    if os.path.exists(basic_model_path):
        print("Basic Model Already Exists!\nWe're loading it...")
        model_basic.load_state_dict(torch.load(basic_model_path))
    if os.path.exists(cnn_model_path):
        print("CNN Model Already Exists!\nWe're loading it...")
        model_cnn.load_state_dict(torch.load(cnn_model_path))

    #optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer1 = optim.SGD(model_basic.parameters(), lr=learning_rate, momentum=0.9)
    optimizer2 = optim.SGD(model_cnn.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.MSELoss()

    #TRAIN MODEL
    num_epochs = 10
    mp.set_start_method('spawn')
    result_queue = mp.Queue()
    processes = []
    proc1 = mp.Process(target=train_model, args=(model_basic, optimizer1, criterion, 'cuda:1', num_epochs, trainloader, validloader, testloader, result_queue))
    proc1.start()
    processes.append((proc1, []))
    proc2 = mp.Process(target=train_model, args=(model_cnn, optimizer2, criterion, 'cuda:1', num_epochs, trainloader, validloader, testloader, result_queue))
    proc2.start()
    processes.append((proc2, []))

    #res = []
    for proc, res in processes:
        proc.join()

    #We're making the assumption that basic neural network always finishes first.
    #This will certainly be the case unless we change the architectures.
    v_epochs = [i for i in range(num_epochs)]

    basic_train_loss = result_queue.get()
    basic_val_loss   = result_queue.get()
    basic_train_acc  = result_queue.get()
    basic_val_acc    = result_queue.get()

    cnn_train_loss   = result_queue.get()
    cnn_val_loss     = result_queue.get()
    cnn_train_acc    = result_queue.get()
    cnn_val_acc      = result_queue.get()

    #PLOT RESULTS
    plt.figure()
    plt.plot(v_epochs, basic_train_loss, label='basic train')
    plt.plot(v_epochs, basic_val_loss,   label='basic validation')
    plt.plot(v_epochs, cnn_train_loss,   label='cnn train')
    plt.plot(v_epochs, cnn_val_loss,     label='cnn validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('4 DROIDS (SHUFFLED HEXES)')
    plt.legend()
    plt.grid(True)
    plt.savefig('PRETRAINED_10EPOCHS_SHUFFLED_DROID4_LOSS.png')

    plt.figure()
    plt.plot(v_epochs, basic_train_acc, label='basic train')
    plt.plot(v_epochs, basic_val_acc,   label='basic validation')
    plt.plot(v_epochs, cnn_train_acc,   label='cnn train')
    plt.plot(v_epochs, cnn_val_acc,     label='cnn validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('4 DROIDS (SHUFFLED HEXES)')
    plt.legend()
    plt.grid(True)
    plt.savefig('PRETRAINED_10EPOCHS_SHUFFLED_DROID4_ACC.png')

    plt.show()
    #print(len(res))

    #print(f'Saving model at: {model_path}\n')
    #torch.save(model.state_dict(), model_path)

    #not pretrained: batch_size = 100, lr = 5*1e-4

    #pretraining: batch_size = 500, lr = 1e-3
    #pretrained:  batch_size = 100, lr = 1e-5