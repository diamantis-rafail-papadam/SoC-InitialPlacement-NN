import numpy as np
from tqdm import tqdm

inputs_npy  = np.load("./DATASET/Inputs.npy")
outputs_npy = np.load("./DATASET/Outputs.npy")

NN_input = []
NN_output = outputs_npy

def hex_padding(hex):
    pad1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    pad2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    layer1 = np.concatenate((pad1, hex[0:4], pad2),   axis=0)
    layer2 = np.concatenate((pad1, hex[4:9], pad1),  axis=0)
    layer3 = np.concatenate((hex[9:15], pad1),       axis=0)
    layer4 = hex[15:22]
    layer5 = np.concatenate((hex[22:28], pad1),       axis=0)
    layer6 = np.concatenate((pad1, hex[28:33], pad1), axis=0)
    layer7 = np.concatenate((pad1, hex[33:], pad2),   axis=0)
    return np.concatenate((layer1, layer2, layer3, layer4, layer5, layer6, layer7), axis=0)

def num_padding(num):
    pad1 = np.array([0])
    pad2 = np.array([0, 0])
    layer1 = np.concatenate((pad1, num[0:4], pad2),   axis=0)
    layer2 = np.concatenate((pad1, num[4:9], pad1),  axis=0)
    layer3 = np.concatenate((num[9:15], pad1),       axis=0)
    layer4 = num[15:22]
    layer5 = np.concatenate((num[22:28], pad1),       axis=0)
    layer6 = np.concatenate((pad1, num[28:33], pad1), axis=0)
    layer7 = np.concatenate((pad1, num[33:], pad2),   axis=0)
    return np.concatenate((layer1, layer2, layer3, layer4, layer5, layer6, layer7), axis=0)

for it in tqdm(range(len(inputs_npy))):
    row = inputs_npy[it]
    hex_layout, num_layout, placements = np.array_split(row, [37, 74])
    new_hex_layout = []
    new_placements = []

    for i in range(len(num_layout)):
        if num_layout[i] == -1:
            num_layout[i] = 0
        elif num_layout[i] == 0:
            #num_layout[i] = 2
            num_layout[i] = 3 #3%
        elif num_layout[i] == 1:
            #num_layout[i] = 3
            num_layout[i] = 6 #6%
        elif num_layout[i] == 2:
            #num_layout[i] = 4
            num_layout[i] = 8 #8%
        elif num_layout[i] == 3:
            #num_layout[i] = 5
            num_layout[i] = 11 #11%
        elif num_layout[i] == 4:
            #num_layout[i] = 6
            num_layout[i] = 14 #14%
        elif num_layout[i] == 5:
            #num_layout[i] = 8
            num_layout[i] = 14 #14%
        elif num_layout[i] == 6:
            #num_layout[i] = 9
            num_layout[i] = 11 #11%
        elif num_layout[i] == 7:
            #num_layout[i] = 10
            num_layout[i] = 8 #8%
        elif num_layout[i] == 8:
            #num_layout[i] = 11
            num_layout[i] = 6 #6%
        elif num_layout[i] == 9:
            #num_layout[i] = 12
            num_layout[i] = 3 #3%

    for i in range(len(hex_layout)): #one-hot encoding
        if hex_layout[i] == 0:
            new_hex_layout.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            #num_layout[i] = 17 #robber activated with 7, which has 17% chance. UPDATE: it's better to leave this as 0 since there's no collected resource.
        elif hex_layout[i] == 1:
            new_hex_layout.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif hex_layout[i] == 2:
            new_hex_layout.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif hex_layout[i] == 3:
            new_hex_layout.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif hex_layout[i] == 4:
            new_hex_layout.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif hex_layout[i] == 5:
            new_hex_layout.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif hex_layout[i] == 6:
            new_hex_layout.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif hex_layout[i] >= 7 and hex_layout[i] <= 15:
            new_hex_layout.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        else:
            if hex_layout[i] & 0xF == 1:
                new_hex_layout.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif hex_layout[i] & 0xF == 2:
                new_hex_layout.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif hex_layout[i] & 0xF == 3:
                new_hex_layout.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif hex_layout[i] & 0xF == 4:
                new_hex_layout.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            else:
                new_hex_layout.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    for i in range(len(placements)):
        if placements[i] == 0:
            new_placements.append([0, 0, 0, 0])
        elif placements[i] == 1:
            new_placements.append([1, 0, 0, 0])
        elif placements[i] == 2:
            new_placements.append([0, 1, 0, 0])
        elif placements[i] == 3:
            new_placements.append([0, 0, 1, 0])
        elif placements[i] == 4:
            new_placements.append([0, 0, 0, 1])

    new_hex_layout = np.array(new_hex_layout)
    new_hex_layout = hex_padding(new_hex_layout)
    new_hex_layout = new_hex_layout.flatten()
    num_layout     = num_padding(num_layout)
    new_placements = np.array(new_placements)
    new_placements = new_placements.flatten()
    
    #View the data:
    if it == 0:
        print(f'Number Layout Data:    {num_layout.shape}')
        print(num_layout.reshape(7, 7))
        print(f'Hex Layout Data: {new_hex_layout.shape}')
        print(new_hex_layout.reshape(7, 7, 13))
        print(f'Initial Placement Data: {new_placements.shape}')
        print(new_placements.reshape(54 + 72, 4))
        print(len(placements))
        
    NN_input.append(np.concatenate((new_hex_layout, num_layout, new_placements), axis=0))

NN_input = np.array(NN_input)
print(NN_input.shape)
np.save("./DATASET/Preprocessed_Inputs", NN_input)
np.save("./DATASET/Preprocessed_Outputs", NN_output)
