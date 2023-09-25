import os
from tqdm import tqdm
from collections import defaultdict

dir = './logs'
msg_win = 'Winner'
msg_end = '# -'
largest_size = 0

player_wins = defaultdict(int)
player_score = defaultdict(int)


dir_items = os.listdir(dir)
files_only = [item for item in dir_items if os.path.isfile(os.path.join(dir, item))]
total_files = len(files_only)

for i in tqdm(range(len(files_only))):
    filename = files_only[i]

    with open(dir + "/" + filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if msg_end in line:
                v = line.strip().split(': ')
                score = int(v[1][-2:])
                last = v[-1].split(', ')
                bot = last[0]

                largest_size = max(largest_size, len(bot))
                player_score[bot] += score
                if last[-1] == msg_win:
                    player_wins[bot] += 1
    
for key in sorted(player_wins.keys()):
    print(f'{key}:{(largest_size - len(key))*" "} {round(player_wins[key] / total_files, 2)} {round(player_score[key] / total_files, 2)}')