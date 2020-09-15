import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from data.dataloader import Batch_Balanced_Dataset, DataLoader
from model import OcrModel
import numpy as np
from label_helper import CTCLabelConverter
import argparse
import anyconfig
import munch
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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

    model_path = os.path.join(cfg.saved_models_path, cfg.model_name)
    best_model_path = os.path.join(os.path.join(model_path, "best"), "best_model.h5")

    dim = 1
    if cfg.rgb:
        dim = 3
    input_shape = (None, cfg.imgH, cfg.imgW, dim) 

    converter = CTCLabelConverter(cfg.character)
    num_class = len(converter.character)
    model = OcrModel(cfg, num_class)
    model.build(input_shape)
    model.load_weights(best_model_path)
    
    total = 0
    total_test = 0
    true_positive = 0
    for dataset_name in os.listdir(cfg.eval_data):
        local_tp = 0
        local_total = 0
        
        dataLoader = DataLoader(cfg.eval_data, dataset_name, cfg, type_data='eval')
        
        while True:
            images, labels = dataLoader.get_batch()
            if images is None or labels is None:
                break
            # text, _ = converter.encode(labels, batch_max_length=cfg.batch_max_length)
            # indices, values, dense_shape = sparse_tuple_from(text)
            # sparse_labels = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

            predictions = model(images, training=False)
            
            predictions = np.argmax(predictions, axis=2)
            pred_words = converter.decode(predictions, [cfg.batch_max_length + 1] * cfg.batch_size)
            for label, pred_word in zip(labels, pred_words):
                if pred_word == label:
                    local_tp += 1
        total += len(dataLoader)
        true_positive += local_tp
        print("*"*80)
        print("Evaluating on", dataset_name)
        print("True positive:", local_tp)
        print("Number of samples:", len(dataLoader))
        print("Accuracy:", local_tp/len(dataLoader))
        print("*"*80)

    print("SUMMARY")
    print("True positive:", true_positive)
    print("Total:", total)
    print("Total test:", total_test)

    print("Accuracy:", true_positive/total)
        
