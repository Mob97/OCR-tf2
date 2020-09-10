import sys
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import anyconfig
import munch
import numpy as np
import cv2
import tensorflow as tf
import datetime

from data.dataloader import Batch_Balanced_Dataset
from model import OcrModel
from label_helper import CTCLabelConverter
from label_helper import sparse_tuple_from
# tf.keras.backend.set_floatx()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dataset = Batch_Balanced_Dataset(cfg)
        self.valid_dataset = Batch_Balanced_Dataset(cfg, train=False)
        self.train_loss_avg = tf.keras.metrics.Mean("train_loss_avg")
        self.val_loss_avg = tf.keras.metrics.Mean("val_loss_avg")
        self.converter = CTCLabelConverter(cfg.character)
        self.num_class = len(self.converter.character)
        self.model = OcrModel(cfg, self.num_class)
        
        self.loss_func = tf.nn.ctc_loss

        config_log = '  Options: \n'
        for k, v in cfg.items():
            config_log += f'{str(k)}: {str(v)}\n'

        

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg.learning_rate,
            decay_steps=10000,
            decay_rate=0.96,
            staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model_path = os.path.join(cfg.saved_models_path, cfg.model_name)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.summary_dir = os.path.join(os.path.join(model_path, current_time), "logs")
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        train_log_dir = os.path.join(self.summary_dir, "train")
        valid_log_dir = os.path.join(self.summary_dir, "val")
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint = tf.train.Checkpoint(
            feature_extraction=self.model.feature_extraction,
            sequence_modeling=self.model.sequence_modeling,
            decoder=self.model.decoder,
            optimizer=self.optimizer
        )

        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=checkpoint, 
                                                        directory=checkpoint_dir, 
                                                        max_to_keep=5, 
                                                        checkpoint_name='ocr_model'
                                                        )
        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        self.write_log(''.join([config_log, self.train_dataset.get_data_information(), self.valid_dataset.get_data_information(), "START TRAINING"]))


    @tf.function
    def train_step(self, images, sparse_labels):    
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            logit_length = tf.fill([tf.shape(predictions)[0]], tf.shape(predictions)[1])
            loss = self.loss_func(
                labels=sparse_labels, 
                logits=predictions, 
                label_length=None,
                logit_length=logit_length,
                blank_index=num_class-1,
                logits_time_major=False
            )        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return tf.reduce_mean(loss)


    @tf.function
    def test_step(self, images, sparse_labels):
        predictions = self.model(images, training=False)
        logit_length = tf.fill([tf.shape(predictions)[0]], tf.shape(predictions)[1])
        loss = self.loss_func(
            labels=sparse_labels, 
            logits=predictions, 
            label_length=None,
            logit_length=logit_length,
            blank_index=num_class-1,
            logits_time_major=False
        )        
        return loss, predictions

    def train(self):
        for i in range(self.cfg.num_step):
            images, labels = self.train_dataset.get_batch()
            # print(images.shape)
            text, _ = self.converter.encode(labels, batch_max_length=cfg.batch_max_length)
            indices, values, dense_shape = sparse_tuple_from(text)
            sparse_labels = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
            train_loss = self.train_step(images, sparse_labels)
            self.train_loss_avg(train_loss)
            print("Train loss avg:", self.train_loss_avg.result().numpy())
            if i % self.cfg.num_iters_4_valid == 0:
                self.val_loss_avg.reset_states()
                val_loss, predictions = self.test_step(images, sparse_labels)
                predictions = np.argmax(predictions, axis=2)
                pred_words = self.converter.decode(predictions, [self.cfg.batch_max_length] * self.cfg.batch_size)
                true_positive = 0
                total = 0
                for label, pred_word in zip(labels, pred_words):
                    total += 1
                    if pred_word == label:
                        true_positive += 1
                val_accuracy = true_positive/total
                self.val_loss_avg(val_loss)
                log_content = '-' * 80 + '\n'
                log_content += "After Iteration %d:\n" %(i)
                log_content += f"----Train loss: {self.train_loss_avg.result().numpy()}\n"
                log_content += f"----Valid loss: {self.val_loss_avg.result().numpy()}\n"
                log_content += f"----Valid accuracy: {val_accuracy}\n"                

                with self.train_summary_writer.as_default():
                    tf.summary.scalar("Train loss", self.train_loss_avg.result().numpy(), step=i // self.cfg.num_iters_4_valid)
                    self.train_summary_writer.flush()
                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("Valid loss", self.val_loss_avg.result().numpy(), step=i // self.cfg.num_iters_4_valid)
                    tf.summary.scalar("Valid accuracy", val_accuracy, step=i // self.cfg.num_iters_4_valid)
                    self.valid_summary_writer.flush()
                save_path = self.ckpt_manager.save()
                log_content += f"The model has been saved at {save_path}\n"
                log_content += '-' * 80
                print(log_content)
                self.write_log(log_content)
                self.train_loss_avg.reset_states()
            
    def write_log(self, message):        
        log = open(os.path.join(self.summary_dir, 'log_training.txt'), 'a')
        log.write(message + '\n')
        log.close()

def parse_args(argv):
	parser = argparse.ArgumentParser(description='Train face network')
	parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config_base.yaml')
	parser.add_argument('--gpu', type=str, help='choose gpu (0,1,2,...) or cpu(-1)', default='0')
	args = parser.parse_args(argv)

	return args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg = anyconfig.load(args.config_path)
    cfg.update(vars(args))
    cfg = munch.munchify(cfg)

    train = Trainer(cfg)
    train.train()












