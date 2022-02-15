import tensorflow as tf
import get_data
import untils

question_vocab, answer_vocab, question_tokens_ids, answer_tokens_ids = get_data.get_dataset()
maxlen = 5
print('question_vocab:', len(question_vocab))
print('answer_vocab:', len(answer_vocab))
print('question_tokens_ids:', len(question_tokens_ids))
print('answer_tokens_ids:', len(answer_tokens_ids))


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def generator(batch_size=32):
    batch_num = len(question_tokens_ids) // batch_size

    while 1:
        for i in range(batch_num):
            start_num = batch_size * i
            end_num = batch_size * (i + 1)
            question_batch = question_tokens_ids[start_num:end_num]
            answer_batch = answer_tokens_ids[start_num:end_num]
            answer_batch_inp = []
            answer_batch_tar = []
            for line in answer_batch:
                answer_batch_inp.append(line[:-1])
                answer_batch_tar.append(line[1:])
            question_batch = tf.keras.preprocessing.sequence.pad_sequences(question_batch,
                                                                           maxlen=maxlen,
                                                                           padding='post',
                                                                           truncating='post')
            answer_batch_inp = tf.keras.preprocessing.sequence.pad_sequences(answer_batch_inp,
                                                                             maxlen=maxlen,
                                                                             padding='post',
                                                                             truncating='post')
            answer_batch_tar = tf.keras.preprocessing.sequence.pad_sequences(answer_batch_tar,
                                                                             maxlen=maxlen,
                                                                             padding='post',
                                                                             truncating='post')

            answer_batch_tar = label_smoothing(tf.one_hot(answer_batch_tar, len(answer_vocab)))

            yield (question_batch, answer_batch_inp), answer_batch_tar


encoder_input = tf.keras.Input(shape=(None,))
decoder_input = tf.keras.Input(shape=(None,))
output = untils.Transformer(len(question_vocab), len(answer_vocab))([encoder_input, decoder_input])
model = tf.keras.Model((encoder_input, decoder_input), output)
print(model.summary())
model.compile(tf.optimizers.Adam(1e-4),
              tf.losses.categorical_crossentropy,
              metrics=["accuracy"])
batch_size = 16
model.load_weights("./saver/model")
model.fit(generator(batch_size), steps_per_epoch=len(answer_tokens_ids) // batch_size, epochs=1000, verbose=2)
model.save_weights("./saver/model")
