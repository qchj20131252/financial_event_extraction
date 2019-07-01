from tqdm import *
import json
import pandas as pd
import numpy as np
import os
from tqdm import *
from pyhanlp import *


maxlen = 256

# 读取数据，排除“其他”类型
D = pd.read_csv('../data/event_type_entity_extract_train.csv', encoding='utf-8', header=None)
D = D[D[2] != u'其他']
D = D[D[1].str.len() <= maxlen]

if not os.path.exists('../dict/classes.json'):
    id2class = dict(enumerate(D[2].unique()))
    class2id = {j: i for i, j in id2class.items()}
    json.dump([id2class, class2id], open('../dict/classes.json', mode='w', encoding='utf-8'), indent=4, ensure_ascii=False)
else:
    id2class, class2id = json.load(open('../dict/classes.json', mode='r', encoding='utf-8'))

if not os.path.exists('../data/train_data.json'):
    CRFnewSegment = HanLP.newSegment("crf")

    train_data = []
    for id, text, type, subject in tqdm(zip(D[0], D[1], D[2], D[3])):
        dic = {'id': id, 'text': text.strip(), 'type': type, 'subject': subject}
        postag = []
        for word_pos in CRFnewSegment.seg(text.strip()):
            split_list = str(word_pos).split('/')
            pos = split_list[-1]
            word = split_list[0]
            if len(split_list) > 2:
                for i in split_list[1:-1]:
                    word += i
            postag.append({'word': word, 'pos': pos})
        dic['postag'] = postag
        train_data.append(dic)
    json.dump(train_data, open('../data/train_data.json', mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)
else:
    train_data = json.load(open('../data/train_data.json', mode='r', encoding='utf-8'))




