import json
from tqdm import *

train_data = json.load(open('../data/train_data.json', mode='r', encoding='utf-8'))

char_dict = {}

for dic in tqdm(train_data):
    text = dic['text']
    for char in text:
        char_dict[char] = char_dict.get(char, 0) + 1

sorted_list = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
char_list = [char[0] for char in sorted_list]
char_list.insert(0, '<UNK>')
char2id = {i: j for i, j in enumerate(char_list)}
id2char = {j: i for i, j in enumerate(char_list)}

json.dump([char2id, id2char], open('../dict/char_dict', mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)