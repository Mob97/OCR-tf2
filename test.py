import anyconfig
import munch
import numpy as np
from data.dataloader import Batch_Balanced_Dataset
import cv2

cfg = anyconfig.load("config.yaml")
cfg = munch.munchify(cfg)
print(cfg.select_data)

dataLoader = Batch_Balanced_Dataset(cfg)
a, b = dataLoader.get_batch()
# print(a[0])
cv2.imwrite('test.png', (a[3]*127.5+127.5).astype('uint8'))
print('.........', a.shape, len(b))
print(b[0])