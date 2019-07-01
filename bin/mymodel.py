import paddle
import paddle.fluid as fluid

def db_lstm(char):
    char_param = fluid.ParamAttr(name="charemb", trainable=True)
    char_embedding = fluid.layers.embedding(
        input=char,
        size=[3604, 512],
        dtype='float32',
        is_distributed=True,
        is_sparse=True,
        param_attr=char_param)

    lstm = fluid.layers.dynamic_lstm(input=char_embedding, size=512, candidate_activation='relu',
                                     gate_activation='sigmoid', cell_activation="sigmoid")
    feature_out = fluid.layers.fc(input=lstm, size=4, act='tanh')

    return feature_out