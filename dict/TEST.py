import json

label_path = "./label_dict"
label_list = []
with open(label_path, mode='r', encoding='utf-8') as fr:
    for line in fr.readlines():
        label_list.append(line.strip())

label2id = {str(i): label for i, label in enumerate(label_list)}
id2label = {label: i for i, label in enumerate(label_list)}

json.dump([label2id, id2label], open(label_path, mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)

postag_path = "./postag_dict"
postag_list = []

with open(postag_path, mode='r', encoding='utf-8') as fr:
    for line in fr.readlines():
        postag_list.append(line.strip())

postag2id = {str(i): postag for i, postag in enumerate(postag_list)}
id2postag = {postag: i for i, postag in enumerate(postag_list)}

json.dump([postag2id, id2postag], open(postag_path, mode='w', encoding='utf-8'), ensure_ascii=False, indent=4)
