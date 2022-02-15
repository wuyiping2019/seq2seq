#encoder层貌似没有问题，现在问题集中在decoder层

import tensorflow as tf
import get_data
import untils

pinyin_vocab,hanzi_vocab,pinyin_tokens_ids,hanzi_tokens_ids = get_data.get_dataset()

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)

def generator(batch_size=32):
    batch_num = len(pinyin_tokens_ids)//batch_size

    while 1:
        for i in range(batch_num):
            start_num = batch_size*i
            end_num = batch_size*(i+1)

            pinyin_batch = pinyin_tokens_ids[start_num:end_num]
            hanzi_batch = hanzi_tokens_ids[start_num:end_num]

            hanzi_batch_inp = [];hanzi_batch_tar = []
            for line in hanzi_batch:
                hanzi_batch_inp.append(line[:-1])
                hanzi_batch_tar.append(line[1:])

            pinyin_batch = tf.keras.preprocessing.sequence.pad_sequences(pinyin_batch,maxlen=32,padding='post', truncating='post')
            hanzi_batch_inp = tf.keras.preprocessing.sequence.pad_sequences(hanzi_batch_inp,maxlen=32,padding='post', truncating='post')
            hanzi_batch_tar = tf.keras.preprocessing.sequence.pad_sequences(hanzi_batch_tar, maxlen=32, padding='post',truncating='post')

            hanzi_batch_tar = label_smoothing(tf.one_hot(hanzi_batch_tar,4462))

            yield (pinyin_batch,pinyin_batch),hanzi_batch_tar


print("pinyin_vocab大小为:",len(pinyin_vocab)) #pinyin_vocab大小为: 1154
print("hanzi_vocab大小为:",len(hanzi_vocab))   #hanzi_vocab大小为: 4462
"---------------------------------------------------------------------------"
encoder_input = tf.keras.Input(shape=(None,))
decoder_input = tf.keras.Input(shape=(None,))

output = untils.Transformer(encoder_vocab_size=1154, decoder_vocab_size=4462)([encoder_input,decoder_input])
model = tf.keras.Model((encoder_input,decoder_input),output)

model.compile(tf.optimizers.Adam(1e-4),tf.losses.categorical_crossentropy,metrics=["accuracy"])
batch_size = 64

model.fit_generator(generator(batch_size),steps_per_epoch=256//batch_size,epochs=1024)




