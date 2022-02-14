class Config:
    def __init__(self, encoder_size=0,
                 decoder_size=0,
                 sample_count=10000,
                 maxlen=10,
                 batch_size=64,
                 epochs=10,
                 units=32,
                 embedding_size=32,
                 vali_rate=0.2):
        """
        :param maxlen: 输入句子截取的长度
        :param batch_size: 梯度下降样本数量
        :param epochs: 训练次数
        :param units: LSTM中的units参数
        :param embedding_size: Embedding中的output_dim参数
        :param vali_rate: model.fit函数的validation_split参数
        """
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.epochs = epochs
        self.units = units
        self.embedding_size = embedding_size
        self.vali_rate = vali_rate
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.encoder_word2id = {}
        self.encoder_id2word = {}
        self.decoder_word2id = {}
        self.decoder_id2word = {}
        self.sample_count = sample_count
