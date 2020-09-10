import tensorflow as tf

class BidirectionalLSTM(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, num_layers=2):
        super(BidirectionalLSTM, self).__init__()
        # self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, activation='sigmoid', return_sequences=True))
        self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size, return_sequences=True))
        self.linear = tf.keras.layers.Dense(output_size, activation='linear')
        
    @tf.function
    def call(self, input_tensors):
        net = self.rnn(input_tensors)
        net = self.linear(net)
        return net