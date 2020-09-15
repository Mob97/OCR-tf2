import sys
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import anyconfig
import munch
import numpy as np
import cv2
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import datetime
import random

from data.dataloader import Batch_Balanced_Dataset
from model import OcrModel
from label_helper import CTCLabelConverter
from label_helper import sparse_tuple_from

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dataset = Batch_Balanced_Dataset(cfg)
        self.valid_dataset = Batch_Balanced_Dataset(cfg, type_data='valid')
        self.train_loss_avg = tf.keras.metrics.Mean("train_loss_avg")
        self.val_loss_avg = tf.keras.metrics.Mean("val_loss_avg")
        self.converter = CTCLabelConverter(cfg.character)
        self.num_class = len(self.converter.character)
        self.model = OcrModel(cfg, self.num_class)
        dim = 1
        if cfg.rgb:
            dim = 3
        input_size = (cfg.batch_size, cfg.imgH, cfg.imgW, dim)
        self.model.build(input_size)
        print(self.model.summary())
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
        self.best_model_dir = os.path.join(model_path, "best")
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)

        checkpoint = tf.train.Checkpoint(
            feature_extraction=self.model.feature_extraction,
            sequence_modeling=self.model.sequence_modeling,
            decoder=self.model.decoder,
            optimizer=self.optimizer
        )

        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=checkpoint, 
                                                        directory=checkpoint_dir, 
                                                        max_to_keep=2, 
                                                        checkpoint_name='ocr_model'
                                                        )
        if self.ckpt_manager.latest_checkpoint:
            checkpoint.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
        
        self.write_log(''.join([config_log, self.train_dataset.get_data_information(), self.valid_dataset.get_data_information(), ' '*35 + "START TRAINING"]))


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
                blank_index=self.num_class-1,
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
            blank_index=self.num_class-1,
            logits_time_major=False
        )
        predictions_prob = tf.nn.softmax(predictions, axis=2)
        predictions_max_prob = tf.reduce_max(predictions_prob, axis=2)
        return loss, predictions, predictions_max_prob

    def train(self):
        best_val_accuracy = -1
        pointer = 0
        for i in range(self.cfg.num_step):
            if pointer*self.cfg.batch_size > len(self.train_dataset):   
                print("-----------EPOCH END----------------")
                self.train_loss_avg.reset_states()
                pointer = 0
            pointer += 1
            images, labels = self.train_dataset.get_batch()            
            text, _ = self.converter.encode(labels, batch_max_length=cfg.batch_max_length)
            indices, values, dense_shape = sparse_tuple_from(text)
            sparse_labels = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

            train_loss = self.train_step(images, sparse_labels)

            self.train_loss_avg(train_loss)
            print("Train loss avg:", self.train_loss_avg.result().numpy())
            if i % self.cfg.num_iters_4_valid == 0:
                true_positive = 0
                total = 0
                n_showed_sample = 0
                log_content = '-' * 80 + '\n'
                log_content += f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F\n'
                log_content += '-' * 80 + '\n'
                self.val_loss_avg.reset_states()
                while total < len(self.valid_dataset) - 1:    
                                    
                    val_images, val_labels = self.valid_dataset.get_batch()  

                    val_text, _ = self.converter.encode(val_labels, batch_max_length=cfg.batch_max_length)
                    indices, values, dense_shape = sparse_tuple_from(val_text)
                    sparse_labels = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

                    val_loss, predictions, predictions_max_prob = self.test_step(val_images, sparse_labels)
                    self.val_loss_avg(val_loss)
                    predictions = np.argmax(predictions, axis=2)
                    pred_words = self.converter.decode(predictions, [self.cfg.batch_max_length + 1] * self.cfg.batch_size)
                    for label, pred_word, pred_max_prob in zip(val_labels, pred_words, predictions_max_prob):
                        total += 1
                        if pred_word == label:
                            true_positive += 1
                        try:
                            confidence_score = tf.math.cumprod(pred_max_prob, axis=0)[-1]
                        except:
                            confidence_score = 0  # for empty pred case
                        if n_showed_sample < self.cfg.nSample_2_show and random.choices([True, False])[0]:
                            log_content += f'{label:25s} | {pred_word:25s} | {confidence_score:0.4f}\t{str(pred_word == label)}\n'
                            n_showed_sample += 1
                log_content += '-' * 80 + '\n'
                val_accuracy = true_positive/total
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.model.save_weights(os.path.join(self.best_model_dir, "best_model.h5"))
                    log_content += "New best valid accuracy {:.4f}\n".format(best_val_accuracy)
                    log_content += "Saving best accuracy weight to " + os.path.join(self.best_model_dir, "best_model.h5" + "\n")                
                
                log_content += "After Iteration %d:\n" %(i)
                log_content += f"----Train loss: {self.train_loss_avg.result().numpy()}\n"
                log_content += f"----Valid loss: {self.val_loss_avg.result().numpy()}\n"
                log_content += f"----True positive: {true_positive}\n"   
                log_content += f"----Total valid samples: {total}\n"   
                log_content += f"----Valid accuracy: {val_accuracy}\n"        



                with self.train_summary_writer.as_default():
                    tf.summary.scalar("Loss", self.train_loss_avg.result().numpy(), step=i // self.cfg.num_iters_4_valid)
                    self.train_summary_writer.flush()
                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("Loss", self.val_loss_avg.result().numpy(), step=i // self.cfg.num_iters_4_valid)
                    tf.summary.scalar("Accuracy", val_accuracy, step=i // self.cfg.num_iters_4_valid)
                    self.valid_summary_writer.flush()
                save_path = self.ckpt_manager.save()
                log_content += f"The model has been saved at {save_path}\n"
                log_content += '-' * 80
                print(log_content)
                self.write_log(log_content)                
            
    def write_log(self, message):        
        log = open(os.path.join(self.summary_dir, 'log_training.txt'), 'a')
        log.write(message + '\n')
        log.close()

def parse_args(argv):
	parser = argparse.ArgumentParser(description='Train face network')
	parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config_base.yaml')
	args = parser.parse_args(argv)

	return args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    

    cfg = anyconfig.load(args.config_path)
    cfg.update(vars(args))
    cfg = munch.munchify(cfg)

    train = Trainer(cfg)
    train.train()












