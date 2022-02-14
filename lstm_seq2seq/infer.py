import os.path
import sys

import tensorflow as tf
import numpy as np
from Config import Config
from DataProcess import DataProcess
from Seq2seq import model, train, encoder, decoder
import jieba


def decode_sequence(input_seq, config, encoder_model, decoder_model):
    # encoder_states = [state_h, state_c]
    states_value = encoder_model.predict(input_seq)  # list 2个 array 1*rnn_size
    target_seq = np.zeros((1, 1))
    # 目标输入序列 初始为 'BEGIN_' 的 idx
    target_seq[0, 0] = config.decoder_word2id['GO']
    stop = False
    decoded_sentence = ''
    # print([target_seq] + states_value)
    while not stop:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # output_tokens [1*1*9126]   h,c [1*rnn_size]
        sampled_token_idx = np.argmax(output_tokens)
        sampled_word = config.decoder_id2word[sampled_token_idx]
        decoded_sentence += sampled_word
        if sampled_word == 'END' or len(decoded_sentence) > 8:
            stop = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_idx  # 作为下一次预测，输入
        # Update states
        states_value = [h, c]  # 作为下一次的状态输入
    return decoded_sentence


config = Config(sample_count=10000,
                maxlen=5,
                batch_size=64,
                epochs=1000,
                units=8,
                embedding_size=8,
                vali_rate=0.2)
data_process = DataProcess('../data/xiaohuangji50w_nofenci.conv', config)
config, encoder_input_data, decoder_input_data, decoder_output_data = data_process.data_process()
model_ = model(config=config)
print('model-----------')
model_.summary()
_, encoder_state, encoder_model = encoder(config)
print('encoder model-----------')
encoder_model.summary()
_, _, _, decoder_model = decoder(config, encoder_state)
print('decoder model-----------')
decoder_model.summary()
model_file = '../model/lstm_seq2seq.h5'
if os.path.exists(model_file):
    tf.keras.models.load_model(model_file)
else:
    train([config, encoder_input_data, decoder_input_data, decoder_output_data])
# 简单测试 采样
while True:
    input_text = input('question:')
    if input_text == 'exit' or input_text == 'quit':
        sys.exit()
    text_to_translate = jieba.lcut(input_text)
    encoder_input_to_translate = np.zeros(
        (1, config.maxlen),
        dtype=np.float32)
    for t, word in enumerate(text_to_translate):
        # print(word)
        if t > config.maxlen - 1:
            break
        encoder_input_to_translate[0, t] = config.encoder_word2id[word]
    # encoder_input_to_translate [[ids,...,0,0,0,0]]
    print('answer:', decode_sequence(encoder_input_to_translate, config, encoder_model, decoder_model))
