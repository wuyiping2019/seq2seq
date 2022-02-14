from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from Config import Config
from DataProcess import DataProcess
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import jieba


def encoder(config: Config):
    # 编码器
    encoder_inputs = Input(shape=(None,), name='encoder_input')
    encoder_after_embedding = Embedding(input_dim=config.encoder_size,  # 单词个数
                                        output_dim=config.embedding_size,
                                        name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(units=config.units, return_state=True, name='encoder_lstm')
    # return_state: Boolean. Whether to return
    #   the last state in addition to the output.
    _, state_h, state_c = encoder_lstm(encoder_after_embedding)
    encoder_states = [state_h, state_c]  # 思想向量
    return encoder_inputs, encoder_states, Model(encoder_inputs, encoder_states)


def decoder(config: Config, encoder_states):
    # 解码器 这个地方必须使用(None,)
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    decoder_after_embedding = Embedding(input_dim=config.decoder_size,  # 单词个数
                                        output_dim=config.embedding_size,
                                        name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(units=config.units,
                        return_sequences=True,
                        return_state=True,
                        name='decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_after_embedding,
                                                     initial_state=encoder_states)
    decoder_states = [state_h, state_c]
    # 使用 encoder 输出的思想向量初始化 decoder 的 LSTM 的初始状态
    decoder_dense = Dense(config.decoder_size,
                          activation='softmax',
                          name='decoder_dense')
    # 输出词个数,多分类
    decoder_outputs = decoder_dense(decoder_outputs)

    # 以下为输出decoder模型
    # 输入decoder
    decoder_state_input_h = Input(shape=(config.units,))
    decoder_state_input_c = Input(shape=(config.units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs_inf, state_h_inf, state_c_inf = \
        decoder_lstm(decoder_after_embedding,
                     initial_state=decoder_states_inputs)
    # 作为下一次推理的状态输入 h, c
    decoder_states_inf = [state_h_inf, state_c_inf]
    # LSTM的输出，接 FC，预测下一个词是什么
    decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
    return decoder_inputs, \
           decoder_outputs, \
           decoder_states, \
           Model([decoder_inputs] + decoder_states_inputs,
                 [decoder_outputs_inf] + decoder_states_inf)


def model(config: Config):
    encoder_inputs, encoder_states, _ = encoder(config)
    decoder_inputs, decoder_outputs, _, _ = decoder(config, encoder_states)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(inputs, file_path='weights.best.h5', best_path='model.h5'):
    config, encoder_input_data, decoder_input_data, decoder_output_data = inputs
    model_ = model(config)
    # 有一次提升, 则覆盖一次 save_best_only=True
    checkpoint = ModelCheckpoint(file_path, monitor='accuracy', verbose=1,
                                 save_best_only=True, mode='max', save_freq=2)
    callbacks_list = [checkpoint]
    history = model_.fit(x=[encoder_input_data, decoder_input_data], y=decoder_output_data,
                         batch_size=128, epochs=config.epochs, validation_split=0.1,
                         callbacks=callbacks_list)
    model_.save(best_path)
    return history