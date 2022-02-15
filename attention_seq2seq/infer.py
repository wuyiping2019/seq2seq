import jieba
import tensorflow as tf
import get_data
import untils
import numpy as np

question_vocab, answer_vocab, question_tokens_ids, answer_tokens_ids = get_data.get_dataset()
print("question_vocab:", len(question_vocab))
print("answer_vocab:", len(answer_vocab))
print('question_tokens_ids:', len(question_tokens_ids))
print('answer_tokens_ids:', len(answer_tokens_ids))
# [182], [73]
# [608, 735, 732, 759, 90], [923, 608, 175, 994]
from train import maxlen
encoder_input = tf.keras.Input(shape=(None,))
decoder_input = tf.keras.Input(shape=(None,))
output = untils.Transformer(len(question_vocab), len(answer_vocab))(
    [encoder_input, decoder_input]
)
model = tf.keras.Model((encoder_input, decoder_input), output)
model.load_weights("./saver/model")

input_text = input('question:')
question_batch = np.array([[question_vocab.index(item) for item in jieba.lcut(input_text)]])
question_batch = tf.keras.preprocessing.sequence.pad_sequences(question_batch,
                                                               maxlen=maxlen,
                                                               padding='post',
                                                               truncating='post')
answer_batch_inp = np.zeros(shape=(1, maxlen))
answer_batch_inp[0, 0] = answer_vocab.index('GO')
target = 0
for i in range(3):
    print(answer_vocab[29])
    tar = model.predict(x=(question_batch, answer_batch_inp[:, 0:i + 1]))
    print('预测结果:', tf.argmax(tar, axis=-1))
    if i == 2:
        target = tar
        break
    answer_batch_inp[0, i + 1] = tf.argmax(tar, axis=-1)[0, i]
print("----------final----------")
print('final:', tf.argmax(target, axis=-1))
print('answer:', [answer_vocab[index] for index in tf.argmax(target, axis=-1)[0]])
