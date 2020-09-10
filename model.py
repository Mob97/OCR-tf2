import tensorflow as tf

from modules.feature_extraction import Resnet
from modules.sequence_modeling import BidirectionalLSTM


class OcrModel(tf.keras.Model):
    def __init__(self, cfg, num_class):
        super(OcrModel, self).__init__()
        self.cfg = cfg
        if cfg.FeatureExtraction == "ResNet":
            self.feature_extraction = Resnet(cfg.feature_extraction.output_channel) #output_channel, layer_num_list
        if cfg.SequenceModeling == "BiLSTM":
            self.sequence_modeling = BidirectionalLSTM(cfg.sequence_modeling.hidden_size, cfg.sequence_modeling.hidden_size)
        if cfg.Prediction == "CTC":
            self.decoder = tf.keras.layers.Dense(num_class)

    @tf.function
    def call(self, input_tensors, training=False):
        if self.cfg.FeatureExtraction:
            net = self.feature_extraction(input_tensors)
        if self.cfg.SequenceModeling:
            net = self.sequence_modeling(net)
        if self.cfg.Prediction:
            net = self.decoder(net)
        return net

if __name__ == "__main__":
    import anyconfig
    import munch
    import numpy as np
    cfg = anyconfig.load("config.yaml")
    cfg = munch.munchify(cfg)
    model = OcrModel(cfg)
    print(model(np.ones((1, 32, 256, 1), dtype=np.float32)))