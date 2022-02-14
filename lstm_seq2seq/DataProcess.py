import os.path
import pickle
import sys

from Config import *
import jieba
import tensorflow as tf
import numpy as np


class DataProcess:
    def __init__(self, source_file, config: Config):
        self.source_file = source_file
        self.config = config

    def data_process(self):
        # 标识encoder input和decoder input
        flag = True
        # 记录encoder input的单词
        encoder_vocabs = set()
        # 记录decoder input的单词
        decoder_vocabs = set()
        # 初始化需要处理的数据条数
        count = 0
        # 记录encoder文本数据
        encoder_list = []
        # 记录encoder文本转id的数据并按maxlen进行填充
        encoder_list_pad = []
        # 记录decoder文本数据
        decoder_list = []
        # 记录decoder文本转id的数据并按maxlen进行填充
        decoder_list_pad = []
        with open(self.source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # 当数据据条数达到即跳出
                if count == self.config.sample_count * 2:
                    break
                # 以E开头的文本直接逃过
                if line.startswith('E'):
                    pass
                # 以M开头的文本是训练的样本数据
                if line.startswith('M'):
                    line = line.replace('M', '').strip()
                    splits = jieba.lcut(line)
                    # 将encoder文本单词放入encoder_vocabs中
                    # 将decoder文本单词放入decoder_vocabs中
                    for item in splits:
                        if flag:
                            encoder_vocabs.add(item)
                        else:
                            decoder_vocabs.add(item)
                    # 将encoder文本放入encoder_list中
                    # 将decoder文本放入decoder_list中
                    if flag:
                        encoder_list.append(splits)
                        flag = False
                        count += 1
                    else:
                        decoder_list.append(splits)
                        flag = True
                        count += 1
        # decoder input需要添加GO、END、PAD标识,因此在decoder_vocabs中加入
        decoder_vocabs.add('GO')
        decoder_vocabs.add('END')
        decoder_vocabs = ['PAD'] + sorted(list(decoder_vocabs))
        encoder_vocabs = ['PAD'] + sorted(list(encoder_vocabs))
        self.config.encoder_size = len(encoder_vocabs)
        self.config.decoder_size = len(decoder_vocabs)
        self.config.encoder_word2id = {j: i for i, j in enumerate(encoder_vocabs)}
        self.config.encoder_id2word = {i: j for i, j in enumerate(encoder_vocabs)}
        self.config.decoder_word2id = {j: i for i, j in enumerate(decoder_vocabs)}
        self.config.decoder_id2word = {i: j for i, j in enumerate(decoder_vocabs)}
        encoder_input_data = np.zeros(
            (len(encoder_list), self.config.maxlen),
            # 句子数量，         最大输入句子长度
            dtype=np.float32
        )
        decoder_input_data = np.zeros(
            (len(decoder_list), self.config.maxlen),
            # 句子数量，          最大输出句子长度
            dtype=np.float32
        )
        decoder_output_data = np.zeros(
            (len(decoder_list), self.config.maxlen, self.config.decoder_size),
            # 句子数量，          最大输出句子长度,      输出 tokens ids 个数
            dtype=np.float32
        )
        # 统一句子长度
        for line in encoder_list:
            if len(line) > self.config.maxlen:
                encoder_list_pad.append(line[:self.config.maxlen])
            else:
                encoder_list_pad.append((line + ['PAD'] * (self.config.maxlen - len(line))))
        for line in decoder_list:
            if len(line) > self.config.maxlen - 2:
                decoder_list_pad.append(['GO'] + line[:self.config.maxlen - 2] + ['END'])
            else:
                decoder_list_pad.append(['GO'] + line + ['PAD'] * (self.config.maxlen - len(line) - 2) + ['END'])
        for i, (encoder_text, decoder_text) in enumerate(zip(encoder_list_pad, decoder_list_pad)):
            for t, word in enumerate(encoder_text):
                encoder_input_data[i, t] = self.config.encoder_word2id[word]
            for t, word in enumerate(decoder_text):
                decoder_input_data[i, t] = self.config.decoder_word2id[word]
                if t > 0:
                    # 解码器的输出比输入提前一个时间步
                    decoder_output_data[i, t - 1, self.config.decoder_word2id[word]] = 1.
        return self.config, encoder_input_data, decoder_input_data, decoder_output_data


if __name__ == '__main__':
    config = Config(maxlen=10)
    data_process = DataProcess('data/xiaohuangji50w_nofenci.conv', config)
    config, encoder_input_data, decoder_input_data, decoder_output_data = data_process.data_process()
    print(encoder_input_data.shape)  # (10, 5)
    print(decoder_input_data.shape)  # (10, 5)
    print(decoder_output_data.shape)  # (10, 5, 7879)
    # print(encoder_input_data)  # (10, 5)
    # print(decoder_input_data)  # (10, 5)
    # print(decoder_output_data)  # (10, 5, 7879)
    x = encoder_input_data[10]
    y = decoder_input_data[10]
    target = decoder_output_data[10]
    print(config.encoder_size)
    print(config.decoder_size)
    print(x)
    print([config.encoder_id2word[item] for item in x])
    print(y)
    print([config.decoder_id2word[item] for item in y])
