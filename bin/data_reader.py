import json

import os
import sys

class RcDataReader(object):
    def __init__(self,
                 charemb_dict_path,
                 wordemb_dict_path,
                 postag_dict_path,
                 label_dict_path,
                 train_data_path,
                 eval_data_path):
        self._charemb_dict_path = charemb_dict_path
        self._wordemb_dict_path = wordemb_dict_path
        self._postag_dict_path = postag_dict_path
        self._label_dict_path = label_dict_path
        self._train_data_path = train_data_path
        self._eval_data_path = eval_data_path

        self._dict_path_dict = {'charemb_dict': self._charemb_dict_path,
                                'wordemb_dict': self._wordemb_dict_path,
                                'postag_dict': self._postag_dict_path,
                                'label_dict': self._label_dict_path}

        for input_dict in [charemb_dict_path, wordemb_dict_path, postag_dict_path, label_dict_path, train_data_path,
                           eval_data_path]:
            if not os.path.exists(input_dict):
                raise ValueError("%s not found." % input_dict)
                return

        self._feature_dict = {}
        self._feature_dict['charemb_dict'] = self._load_dict_from_file(self._dict_path_dict['charemb_dict'])
        self._feature_dict['postag_dict'] = self._load_dict_from_file(self._dict_path_dict['postag_dict'])
        self._feature_dict['wordemb_dict'] = self._load_dict_from_file(self._dict_path_dict['wordemb_dict'])
        self._feature_dict['label_dict'] = self._load_dict_from_file(self._dict_path_dict['label_dict'])

        self._UNK_IDX = 0



    def _load_dict_from_file(self, dict_path):
        """
        Load vocabulary from file.
        """
        return json.load(open(dict_path, mode='r', encoding='utf-8'))

    def _is_valid_input_data(self, dic):
        """is the input data valid"""
        if "text" not in dic or "postag" not in dic or \
                type(dic["postag"]) is not list:
            return False
        for item in dic['postag']:
            if "word" not in item or "pos" not in item:
                return False
        return True

    def _get_feed_iterator(self, dic, need_input=False, need_label=True):
        # verify that the input format of each line meets the format
        if not self._is_valid_input_data(dic):
            print('Format is error', sys.stderr)
            return None
        sentence = dic['text']
        sentence_char_list = [char for char in sentence]
        sentence_term_list = []
        for postag in dic['postag']:
            word = postag['word']
            for char in word:
                sentence_term_list.append(word)
        sentence_pos_list = []
        for postag in dic['postag']:
            word = postag['word']
            pos = postag['pos']
            sentence_pos_list.append('B-' + pos)
            if len(word) == 2:
                sentence_pos_list.append('E-' + pos)
            elif len(word) > 2:
                for i in word[1: -1]:
                    sentence_pos_list.append('I-' + pos)
                sentence_pos_list.append('E-' + pos)

        sentence_char_slot = [self._feature_dict['charemb_dict'][1].get(c, self._UNK_IDX) for c in sentence_char_list]
        sentence_emb_slot = [self._feature_dict['wordemb_dict'][1].get(w, self._UNK_IDX) \
                for w in sentence_term_list]
        sentence_pos_slot = [self._feature_dict['postag_dict'][1].get(pos, self._UNK_IDX) \
                for pos in sentence_pos_list]
        label_slot = [self._feature_dict['label_dict'][1]['O']] * len(sentence_term_list)
        subject = dic['subject']
        if subject != "NaN":
            index = sentence.find(subject)
            label_slot[index] = self._feature_dict["label_dict"][1]['B-SUB']
            if len(subject) >= 2:
                for i in range(1, len(subject) - 1):
                    label_slot[index + i] = self._feature_dict["label_dict"][1]['I-SUB']
                label_slot[index + len(subject) - 1] = self._feature_dict["label_dict"][1]['E-SUB']
        # verify that the feature is valid
        if len(sentence_emb_slot) == 0 or len(sentence_pos_slot) == 0 or len(sentence_char_list) == 0 \
                or len(label_slot) == 0:
            return None
        # feature_slot = [sentence_emb_slot, sentence_pos_slot]
        feature_slot = [sentence_char_slot, sentence_emb_slot, sentence_pos_slot]
        # feature_slot = [sentence_char_slot]

        input_fields = json.dumps(dic, ensure_ascii=False).encode('utf-8')
        output_slot = feature_slot
        if need_input:
            output_slot = [input_fields] + output_slot
        if need_label:
            output_slot = output_slot + [label_slot]
        return output_slot

    def path_reader(self, data_path, need_input=False, need_label=True):
        """Read data from data_path"""
        self._feature_dict['data_keylist'] = []
        def reader():
            """Generator"""
            if os.path.isdir(data_path):
                input_files = os.listdir(data_path)
                for data_file in input_files:
                    data_file_path = os.path.join(data_path, data_file)
                    for dic in json.load(open(data_file_path.strip(), mode='r', encoding='utf-8')):
                        sample_result = self._get_feed_iterator(dic, need_input, need_label)
                        if sample_result is None:
                            continue
                        yield tuple(sample_result)
            elif os.path.isfile(data_path):
                for dic in json.load(open(data_path.strip(), mode='r', encoding='utf-8')):
                    sample_result = self._get_feed_iterator(dic, need_input, need_label)
                    if sample_result is None:
                        continue
                    yield tuple(sample_result)

        return reader

    def get_train_reader(self, need_input=False, need_label=True):
        """Data reader during training"""
        return self.path_reader(self._train_data_path, need_input, need_label)

    def get_test_reader(self, need_input=True, need_label=True):
        """Data reader during test"""
        return self.path_reader(self._eval_data_path, need_input, need_label)

    def get_dict_size(self, dict_name):
        """Return dict length"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))
        return len(self._feature_dict[dict_name][0])


if __name__ == '__main__':

    # initialize data generator

    data_generator = RcDataReader(
        charemb_dict_path='../dict/char_dict',
        wordemb_dict_path='../dict/word_dict',
        postag_dict_path='../dict/postag_dict',
        label_dict_path='../dict/label_dict',
        train_data_path='../data/train_data.json',
        eval_data_path='../data/train_data.json')

    # prepare data reader
    ttt = data_generator.get_test_reader()
    for index, features in enumerate(ttt()):
        input_sent, word_idx_list, postag_list, p_idx, label_list = features
        print(input_sent)
        print('1st features:', len(word_idx_list), word_idx_list)
        print('2nd features:', len(postag_list), postag_list)
        print('3rd features:', len(p_idx), p_idx)
        print('4th features:', len(label_list), label_list)