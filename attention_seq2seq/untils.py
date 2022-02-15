import tensorflow as tf
import numpy as np

# embedding_size = 256
# n_head = 8
# n_layer = 3
embedding_size = 4
n_head = 4
n_layer = 2


def splite_tensor(tensor):
    shape = tf.shape(tensor)
    tensor = tf.reshape(tensor, shape=[shape[0], -1, n_head, embedding_size // n_head])
    tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
    return tensor


def point_wise_feed_forward_network(d_model=embedding_size):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_model * 4, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def positional_encoding(position=512, d_model=embedding_size):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_decoder_mask(encoder_input, decoder_input):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(encoder_input)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(encoder_input)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(decoder_input)[1])
    dec_target_padding_mask = create_padding_mask(decoder_input)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def build(self, input_shape):
        self.dense_query = tf.keras.layers.Dense(units=embedding_size, activation=tf.nn.relu)
        self.dense_key = tf.keras.layers.Dense(units=embedding_size, activation=tf.nn.relu)
        self.dense_value = tf.keras.layers.Dense(units=embedding_size, activation=tf.nn.relu)
        self.dense = tf.keras.layers.Dense(units=embedding_size, activation=tf.nn.relu)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        super(MultiHeadAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        query, key, value, mask = inputs
        shape = tf.shape(query)

        query_dense = self.dense_query(query)
        key_dense = self.dense_query(key)
        value_dense = self.dense_query(value)

        query_dense = splite_tensor(query_dense)
        key_dense = splite_tensor(key_dense)
        value_dense = splite_tensor(value_dense)

        attention = tf.matmul(query_dense, key_dense, transpose_b=True) / tf.math.sqrt(
            tf.cast(embedding_size, tf.float32))

        attention += (mask * -1e9)
        attention = tf.nn.softmax(attention)

        attention = tf.matmul(attention, value_dense)
        attention = tf.transpose(attention, [0, 2, 1, 3])

        attention = tf.reshape(attention, [shape[0], -1, embedding_size])

        attention = self.dense(attention)
        attention = self.layer_norm((attention + query))

        return attention


class FeedForWard(tf.keras.layers.Layer):
    def __init__(self):
        super(FeedForWard, self).__init__()

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv1D(filters=embedding_size * 4, kernel_size=1, activation=tf.nn.relu)
        self.conv_2 = tf.keras.layers.Conv1D(filters=embedding_size, kernel_size=1, activation=tf.nn.relu)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.ffn = point_wise_feed_forward_network()
        super(FeedForWard, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        output = self.ffn(inputs)  # 这里是仿照论文中的进行调用，效果还行
        # output = self.conv_1(inputs)
        # output = self.conv_2(output)
        # output = tf.keras.layers.Dropout(0.1)(output)
        output = self.layer_norm((inputs + output))
        return output


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EncoderLayer, self).__init__()

    def build(self, input_shape):
        self.multiHeadAttention = MultiHeadAttention()
        self.feedForWard = FeedForWard()
        super(EncoderLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        encoder_embedding, mask = inputs

        encoder_embedding = self.multiHeadAttention([encoder_embedding, encoder_embedding, encoder_embedding, mask])
        encoder_embedding = self.feedForWard(encoder_embedding)

        return encoder_embedding


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DecoderLayer, self).__init__()

    def build(self, input_shape):
        self.self_multiHeadAttention = MultiHeadAttention()
        self.mutual_multiHeadAttention = MultiHeadAttention()
        self.feedForWard = FeedForWard()
        super(DecoderLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        decoder_embedding, encoder_embedding, look_ahead_mask, padding_mask = inputs

        decoder_embedding = self.self_multiHeadAttention(
            [decoder_embedding, decoder_embedding, decoder_embedding, look_ahead_mask])
        decoder_embedding = self.mutual_multiHeadAttention(
            [decoder_embedding, encoder_embedding, encoder_embedding, padding_mask])
        decoder_embedding = self.feedForWard(decoder_embedding)

        return decoder_embedding


class Encoder(tf.keras.layers.Layer):
    def __init__(self, encoder_vocab_size):
        super(Encoder, self).__init__()
        self.encoder_vocab_size = encoder_vocab_size
        self.word_embedding_table = tf.keras.layers.Embedding(encoder_vocab_size, embedding_size)
        self.position_embedding = positional_encoding()

    def build(self, input_shape):
        self.encoderLayers = [EncoderLayer() for _ in range(n_layer)]
        super(Encoder, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        encoder_token_inputs, encoder_mask = inputs
        encoder_embedding = self.word_embedding_table(encoder_token_inputs)
        position_embedding = tf.slice(self.position_embedding, [0, 0, 0], [1, tf.shape(encoder_token_inputs)[1], -1])
        encoder_embedding = encoder_embedding + position_embedding

        for i in range(n_layer):
            encoder_embedding = self.encoderLayers[i]([encoder_embedding, encoder_mask])
        return encoder_embedding


class Decoder(tf.keras.layers.Layer):
    def __init__(self, decoder_vocab_size):
        super(Decoder, self).__init__()
        self.decoder_vocab_size = decoder_vocab_size
        self.word_embedding_table = tf.keras.layers.Embedding(decoder_vocab_size, embedding_size)
        self.position_embedding = positional_encoding()

    def build(self, input_shape):
        self.decoderLayers = [DecoderLayer() for _ in range(n_layer)]

        super(Decoder, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        decoder_token_inputs, encoder_embedding, look_ahead_mask, padding_mask = inputs

        decoder_embedding = self.word_embedding_table(decoder_token_inputs)
        position_embedding = tf.slice(self.position_embedding, [0, 0, 0], [1, tf.shape(decoder_token_inputs)[1], -1])
        decoder_embedding = decoder_embedding + position_embedding

        for i in range(n_layer):
            decoder_embedding = self.decoderLayers[i](
                [decoder_embedding, encoder_embedding, look_ahead_mask, padding_mask])

        return decoder_embedding


class Transformer(tf.keras.layers.Layer):
    def __init__(self, encoder_vocab_size, decoder_vocab_size):
        super(Transformer, self).__init__()
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size

    def build(self, input_shape):
        self.encoder = Encoder(self.encoder_vocab_size)
        self.decoder = Decoder(self.decoder_vocab_size)
        self.final_layer = tf.keras.layers.Dense(self.decoder_vocab_size, tf.nn.softmax)
        super(Transformer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        encoder_input, decoder_input = inputs
        enc_padding_mask, combined_mask, dec_padding_mask = create_decoder_mask(encoder_input, decoder_input)

        # encoder部分
        encoder_embedding = self.encoder([encoder_input, enc_padding_mask])

        # decoder部分
        decoder_embedding = self.decoder([decoder_input, encoder_embedding, combined_mask, dec_padding_mask])

        output = self.final_layer(decoder_embedding)
        return output
