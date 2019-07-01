import json
from tqdm import *

train_data = json.load(open('../data/train_data.json', mode='r', encoding='utf-8'))

word_dict = {}
for dic in tqdm(train_data):
    postag = dic['postag']
    for word_pos in postag:
        word = word_pos['word']
        word_dict[word] = word_dict.get(word, 0) + 1

sorted_list = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
word_list = [word[0] for word in sorted_list]
word_list.insert(0, '<UNK>')

word2id = {i: j for i, j in enumerate(word_list)}
id2word = {j: i for i, j in enumerate(word_list)}

json.dump([word2id, id2word], open('../dict/word_dict', mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)
