import tensorflow as tf
import get_data
import untils

pinyin_vocab, hanzi_vocab, pinyin_tokens_ids, hanzi_tokens_ids = get_data.get_dataset()


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def generator(batch_size=32):
    batch_num = len(pinyin_tokens_ids) // batch_size

    while 1:
        for i in range(batch_num):
            start_num = batch_size * i
            end_num = batch_size * (i + 1)

            pinyin_batch = pinyin_tokens_ids[start_num:end_num]
            hanzi_batch = hanzi_tokens_ids[start_num:end_num]

            hanzi_batch_inp = [];
            hanzi_batch_tar = []
            for line in hanzi_batch:
                hanzi_batch_inp.append(line[:-1])
                hanzi_batch_tar.append(line[1:])

            pinyin_batch = tf.keras.preprocessing.sequence.pad_sequences(pinyin_batch, maxlen=48, padding='post',
                                                                         truncating='post')
            hanzi_batch_inp = tf.keras.preprocessing.sequence.pad_sequences(hanzi_batch_inp, maxlen=48, padding='post',
                                                                            truncating='post')
            hanzi_batch_tar = tf.keras.preprocessing.sequence.pad_sequences(hanzi_batch_tar, maxlen=48, padding='post',
                                                                            truncating='post')

            hanzi_batch_tar = label_smoothing(tf.one_hot(hanzi_batch_tar, 4462))

            yield (pinyin_batch, hanzi_batch_inp), hanzi_batch_tar


print("pinyin_vocab大小为:",len(pinyin_vocab)) #pinyin_vocab大小为: 1154
print("hanzi_vocab大小为:",len(hanzi_vocab))   #hanzi_vocab大小为: 4462

encoder_input = tf.keras.Input(shape=(None,))
decoder_input = tf.keras.Input(shape=(None,))

output = untils.Transformer(1154,4462)([encoder_input,decoder_input])
model = tf.keras.Model((encoder_input, decoder_input), output)
model.load_weights("./saver/model")

(pinyin_batch,hanzi_batch_inp),hanzi_batch_tar = next(generator())

choice = 17

for inp,tar in zip(pinyin_batch,hanzi_batch_inp):

    output = [3]
    for i in range(48):
        predition = model.predict(x=([inp],[output]))
        predition = (tf.argmax(predition,axis=-1))[0]
        output.append(predition.numpy()[-1])

    print(output)
    print(tar)
    print("----------------------------------")


# inp = pinyin_batch[choice]
# output = [3]
# for i in range(48):
#     predition = model.predict(x=([inp],[output]))
#     predition = (tf.argmax(predition,axis=-1))[0]
#     output.append(predition.numpy()[-1])
#     if output[-1] == 2:
#         break
# print(output)

#     output += predition.numpy()


print("----------bad----------")

tar = model.predict(x=(pinyin_batch,hanzi_batch_inp))
print((tf.argmax(tar,axis=-1))[choice])


