import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
import anyconfig
import munch
import numpy as np
from data.dataloader import Batch_Balanced_Dataset
import cv2
from label_helper import CTCLabelConverter
from model import OcrModel
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
    index = 3
    dataLoader = Batch_Balanced_Dataset(cfg, type_data='valid')
    print(len(dataLoader))
    a, b = dataLoader.get_batch()
    image, label = a[index], b[index]
    # print(image)
    print(label)
    # print(a[0])
    cv2.imwrite('test.png', (a[index]*127.5+127.5).astype('uint8'))
    converter = CTCLabelConverter(cfg.character)
    num_class = len(converter.character)

    model_path = os.path.join(cfg.saved_models_path, cfg.model_name)
    best_model_path = os.path.join(os.path.join(model_path, "best"), "best_model.h5")
    
    dim = 1
    if cfg.rgb:
        dim = 3
    input_shape = (None, cfg.imgH, cfg.imgW, dim) 

    model = OcrModel(cfg, num_class)   
    model.build(input_shape)
    model.load_weights(best_model_path)

    pred = model(image[None, ...], training=False)
    print(pred.shape)
    pred_max = np.argmax(pred, axis=2)
    print(pred_max)
    print(pred_max.shape)
    pred_words = converter.decode(pred_max, [cfg.batch_max_length + 1] * cfg.batch_size)
    print(pred_words)

