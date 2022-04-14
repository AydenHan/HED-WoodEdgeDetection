from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import cv2 as cv
import yaml
import os

from data import DataParser
from hed import HED


if __name__ == '__main__':
    # 注：预测需在tf2.3.0版本下进行
    print(tf.__version__)
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # prepare data
    with open('cfg.yml') as file:
        cfg = yaml.load(file)
    dataParser = DataParser(cfg)

    hed = HED(tf.keras.Input((cfg['height'], cfg['width'], cfg['channel'])), cfg['weight_decay_ratio'])
    hed.hed_cnn()
    hed.model.load_weights(os.path.join(cfg['model_weights_path'], 'model.h5'))

    input = [46]
    y = hed.model(dataParser.get_single_test(input), training=False)

    for i in range(len(input)):
        rst = (np.squeeze(y[5][i].numpy()) * 255).astype(np.uint8)
        cv.imwrite(str(i)+'gray.png', rst)
        cv.imwrite(str(i)+'binary.png', 255 * (rst > 220))
