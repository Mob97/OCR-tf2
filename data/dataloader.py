import os
from data.lmdb import LmdbDataset
from PIL import Image
import random
import math
import numpy as np


class Batch_Balanced_Dataset(object):
    def __init__(self, config, train=True):               
        self.data_loader_list = []
        self.batch_size_list = []
        
        if train:
            type_data = 'train'
            data_path = config.train_data
            select_data = config.select_train_data.split('-')
            batch_ratio = config.batch_ratio_train.split('-')    
        else:
            type_data = 'valid'
            data_path = config.valid_data
            select_data = config.select_val_data.split('-')
            batch_ratio = config.batch_ratio_val.split('-')    
        ###LOG
        self.log_content = '-' * 80 + '\n'
        self.log_content += f'dataset_root: {data_path}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}\n'
        for selected_d, batch_ratio_d in zip(select_data, batch_ratio):
            batch_size = max(round(config.batch_size * float(batch_ratio_d)), 1)
            dataLoader = DataLoader(data_path, selected_d, config, type_data)
            self.data_loader_list.append(dataLoader)
            self.batch_size_list.append(batch_size)
            self.log_content += f'num total samples of {selected_d}({type_data}): {dataLoader.dataset.nSamples}\n'
        self.log_content += '-' * 80 + '\n'

    def get_data_information(self):
        return self.log_content

    def get_batch(self):
        image_list = []
        label_list = []
        for data_loader, batch_size in zip(self.data_loader_list, self.batch_size_list):
            image, label = data_loader.get_batch(batch_size)
            image_list.append(image)
            label_list = label_list + list(label)
        return np.concatenate(image_list, axis=0), tuple(label_list)

class DataLoader():
    def __init__(self, root, select_data, config, type_data='train'):
        self.config = config
        self.dataset = LmdbDataset(os.path.join(root, select_data), config)
        self._AlignCollate = AlignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio_with_pad=config.PAD)
        self.type_data = type_data
        self.pointer = 0
        self.data_index = list(range(0, self.dataset.nSamples))
        if type_data == 'train':
            random.shuffle(self.data_index)

    def __len__(self):
        return self.dataset.nSamples

    def get_batch(self, batch_size=None):
        if not batch_size:
            batch_size=self.config.batch_size
        samples = []
        labels = []
        numSamples = self.dataset.nSamples

        if self.type_data == 'train' and (self.pointer + 1) * batch_size >= self.dataset.nSamples:
            self.pointer = 0            
            random.shuffle(self.data_index)
        upper = min((self.pointer + 1) * batch_size, self.dataset.nSamples - 1)
        batch_indexes = self.data_index[self.pointer * batch_size: upper]
        self.pointer += 1

        batch = [self.dataset[x] for x in batch_indexes]
        batch_img, labels = self._AlignCollate(batch)
        for img, label in zip(batch_img, labels):
            img = np.asarray(img)
            img = (img - 127.5) / 127.5
            if len(img.shape) == 2:
                img = img[..., None]
            samples.append(img)
        return np.array(samples), labels

class NormalizePAD(object):
    
    def __init__(self, max_size, PAD_type='right'):
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[1] / 2)
        self.PAD_type = PAD_type

    def __call__(self, image):
        # img = np.subtract(img, 125)
        # img = np.divide(img, 125)
        w, h = image.size
        img = np.asarray(image)
        img = img[..., None]
        Pad_img = np.zeros(self.max_size)
        Pad_img[:, :w, :] = img  # right pad
        if self.max_size[1] != w:  # add border Pad
            last_pixels = np.expand_dims(img[:, w-1, :], axis=1)
            tmp_list = [last_pixels] * (self.max_size[1] - w)
            Pad_img[:, w:, :] = np.concatenate(tmp_list, axis=1)
        Pad_img = np.squeeze(Pad_img)
        return Image.fromarray(Pad_img.astype(np.uint8))


class AlignCollate():
    
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)
        # images, labels = batch

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            # print((self.imgH, resized_max_w, input_channel))
            transform = NormalizePAD((self.imgH, resized_max_w, input_channel))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            # batch_img = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
            batch_img = np.concatenate([np.expand_dims(t, 0) for t in resized_images], 0)
        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            batch_img = np.concatenate([np.expand_dims(t, 0) for t in image_tensors], 0)

        return batch_img, labels

class ResizeNormalize(object):
    
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        # img = np.subtract(img, 125)
        # img = np.divide(img, 125)
        return img


if __name__ == '__main__':
    import anyconfig
    import munch
    import numpy as np
    cfg = anyconfig.load("config.yaml")
    cfg = munch.munchify(cfg)
    print(cfg.select_data)
    dataLoader = Batch_Balanced_Dataset(cfg)
    a, b = dataLoader.get_batch()
    cv2.imwrite('test.png', a[0])
    print('.........', a.shape, len(b))
    print(b[0])