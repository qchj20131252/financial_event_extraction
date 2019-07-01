import paddle
import paddle.fluid as fluid

def db_lstm(data_reader, char, word, postag):
    hidden_dim = 512
    depth = 4
    label_dict_len = 4

    char_param = fluid.ParamAttr(name="charemb", trainable=True)
    char_embedding = fluid.layers.embedding(
        input=char,
        size=[3604, 25],
        dtype='float32',
        is_distributed=True,
        is_sparse=True,
        param_attr=char_param)


    word_param = fluid.ParamAttr(name="wordemb", trainable=True)
    word_embedding = fluid.layers.embedding(
        input=word,
        size=[42388, 128],
        dtype='float32',
        is_distributed=True,
        is_sparse=True,
        param_attr=word_param)

    postag_param = fluid.ParamAttr(name="posemb", trainable=True)
    postag_embedding = fluid.layers.embedding(
        input=postag,
        size=[444, 25],
        dtype='float32',
        is_distributed=True,
        is_sparse=True,
        param_attr=postag_param)

    emb_layers = [char_embedding, word_embedding, postag_embedding]
    # emb_layers = [char_embedding]

    hidden_0_layers = [fluid.layers.fc(input=emb, size=hidden_dim, act='tanh') for emb in emb_layers]

    hidden_0 = fluid.layers.concat(input=hidden_0_layers)

    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = fluid.layers.concat(input=[
            fluid.layers.fc(input=input_tmp[0], size=hidden_dim, act='tanh'),
            fluid.layers.fc(input=input_tmp[1], size=hidden_dim, act='tanh')
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    # output
    feature_out = fluid.layers.concat(input=[
        fluid.layers.fc(input=input_tmp[0], size=label_dict_len, act='tanh'),
        fluid.layers.fc(input=input_tmp[1], size=label_dict_len, act='tanh')
    ])

    return feature_out


