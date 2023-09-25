import numpy as np
import os
from tqdm import tqdm
import h5py

dir = './logs'
msg1 = 'SOCBoardLayout'
msg2 = 'SOCPutPiece'
msg3 = 'SOCGameMembers'
msg_srv = 'all' #only needed when we store client messages in SOCLOG files.
msg_win = 'Winner'
msg_end = '# -'
input  = []
output = []
counter = 0

settlement_coords = ['23', '25', '27', '32', '34', '36', '38', '43', '45', '47', '49', '52', '54', '56', '58', '5a', '63', '65', '67', '69', '6b', '72', '74', '76', '78', '7a', '7c', '83', '85', '87', '89', '8b', '8d', '94', '96', '98', '9a', '9c', 'a5', 'a7', 'a9', 'ab', 'ad', 'b6', 'b8', 'ba', 'bc', 'c7', 'c9', 'cb', 'cd', 'd8', 'da', 'dc']
road_coords       = ['22', '23', '24', '25', '26', '27', '32', '34', '36', '38', '42', '43', '44', '45', '46', '47', '48', '49', '52', '54', '56', '58', '5a', '62', '63', '64', '65', '66', '67', '68', '69', '6a', '6b', '72', '74', '76', '78', '7a', '7c', '83', '84', '85', '86', '87', '88', '89', '8a', '8b', '8c', '94', '96', '98', '9a', '9c', 'a5', 'a6', 'a7', 'a8', 'a9', 'aa', 'ab', 'ac', 'b6', 'b8', 'ba', 'bc', 'c7', 'c8', 'c9', 'ca', 'cb', 'cc']

settl_map = {key: i for i, key in enumerate(settlement_coords)}
road_map  = {key: i for i, key in enumerate(road_coords)}

def SOCBoardLayout_getLayouts(board_layout):
    layout = board_layout.split('|')
    if(len(layout) < 2): #corrupted file
        return np.array([]), np.array([])
    data_hex = layout[1][layout[1].index('{') + 2 : layout[1].index('}') - 1]
    data_number = layout[2][layout[2].index('{') + 2 : layout[2].index('}') - 1]
    return np.array([num.strip() for num in data_hex.split(' ')], dtype=int), np.array([num.strip() for num in data_number.split(' ')], dtype=int)

def SOCPutPiece_getPlayers(placed_pieces):
    p0_pieces = []
    p1_pieces = []
    p2_pieces = []
    p3_pieces = []
    for piece in placed_pieces:
        if "playerNumber=0" in piece:
            p0_pieces.append(piece)
        elif "playerNumber=1" in piece:
            p1_pieces.append(piece)
        elif "playerNumber=2" in piece:
            p2_pieces.append(piece)
        else:
            p3_pieces.append(piece)
    return p0_pieces, p1_pieces, p2_pieces, p3_pieces

def SOCPutPiece_getTypes(placed_pieces):
    roads  = []
    settl  = []
    cities = []
    for piece in placed_pieces:
        if "pieceType=0" in piece:
            roads.append(piece)
        elif "pieceType=1" in piece:
            settl.append(piece)
        elif "pieceType=2" in piece:
            cities.append(piece)
    return roads, settl, cities

def SOCPutPiece_getCoords(placed_pieces):
    coords = []
    for piece in placed_pieces:
        coords.append(piece[-3:][:2])
    return coords

def SOCGameMembers_getPlayers(line):
    info = line.split('|')
    if(len(info) < 2): #corrupted file
        return np.array([]), np.array([])
    members = info[1][info[1].index('[') + 1 : info[1].index(']')]
    return np.array([pl.strip() for pl in members.split(', ')], dtype=str)

dir_items = os.listdir(dir)
files_only = [item for item in dir_items if os.path.isfile(os.path.join(dir, item))]
total_files = len(files_only)

for i in tqdm(range(len(files_only))):
    filename = files_only[i]
    counter += 1
    #print(f'Reading file {counter} / {total_files}')

    #Variables for current file
    board_layout = ""
    winner_line = ""
    placed_pieces = []
    curr_players = []
    score = [0, 0, 0, 0]

    with open(dir + "/" + filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if msg1 in line:
                board_layout = line.strip()
            elif msg2 in line and msg_srv in line:
                placed_pieces.append(line)
            elif msg_end in line:
                #print(line)
                line_list = line.strip().split(':')
                pn = int(line_list[0][-1])
                sc = int(line_list[1].split()[-1])
                score[pn] = sc
    
    #SOCBoardLayout stuff
    hex_layout, number_layout = SOCBoardLayout_getLayouts(board_layout)
    if(len(hex_layout) == 0 or len(number_layout) == 0): #corrupted file
        continue

    #SOCPutPiece phase stuff
    initial_placements = placed_pieces[:16] #used to speed up log extraction

    p0_pieces, p1_pieces, p2_pieces, p3_pieces = SOCPutPiece_getPlayers(initial_placements) #we can use "placed_pieces" if we're interested in more than initial phase.

    p0_roads, p0_settl, p0_cities = SOCPutPiece_getTypes(p0_pieces)
    p1_roads, p1_settl, p1_cities = SOCPutPiece_getTypes(p1_pieces)
    p2_roads, p2_settl, p2_cities = SOCPutPiece_getTypes(p2_pieces)
    p3_roads, p3_settl, p3_cities = SOCPutPiece_getTypes(p3_pieces)

    #Initial roads and settlements for NN evaluation.
    p0_settl_init = p0_settl[:2]
    p0_roads_init = p0_roads[:2]
    p1_settl_init = p1_settl[:2]
    p1_roads_init = p1_roads[:2]
    p2_settl_init = p2_settl[:2]
    p2_roads_init = p2_roads[:2]
    p3_settl_init = p3_settl[:2]
    p3_roads_init = p3_roads[:2]

    initial_settlements = [0 for _ in range(54)]
    initial_roads       = [0 for _ in range(72)]

    for road in SOCPutPiece_getCoords(p0_roads_init):
        initial_roads[road_map[road]] = 1
    for road in SOCPutPiece_getCoords(p1_roads_init):
        initial_roads[road_map[road]] = 2
    for road in SOCPutPiece_getCoords(p2_roads_init):
        initial_roads[road_map[road]] = 3
    for road in SOCPutPiece_getCoords(p3_roads_init):
        initial_roads[road_map[road]] = 4

    for settl in SOCPutPiece_getCoords(p0_settl_init):
        initial_settlements[settl_map[settl]] = 1
    for settl in SOCPutPiece_getCoords(p1_settl_init):
        initial_settlements[settl_map[settl]] = 2
    for settl in SOCPutPiece_getCoords(p2_settl_init):
        initial_settlements[settl_map[settl]] = 3
    for settl in SOCPutPiece_getCoords(p3_settl_init):
        initial_settlements[settl_map[settl]] = 4

    curr = np.concatenate((hex_layout, number_layout, np.array(initial_roads), np.array(initial_settlements)), axis=0)
    if len(curr) != 200: #corrupted file
        continue
    input.append(curr)
    output.append(score)

NN_input  = np.array(input)
NN_output = np.array(output)

print(f'First row of the input data looks like:\n{NN_input[0]}\n')
print(f'First row of the output data looks like:\n{NN_output[0]}\n')

print('Writing to DATASET:')
print(f'Input:  {NN_input.shape}')
print(f'Output: {NN_output.shape}')

np.save("./DATASET/Inputs", NN_input)
np.save("./DATASET/Outputs", NN_output)

#with h5py.File("./DATASET/dataset.h5", "w") as file:
#    file.create_dataset("inputs", data=NN_input)
#    file.create_dataset("outputs", data=NN_output)
