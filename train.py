from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import yaml
import os

from data import DataParser
from hed import HED, cross_entropy_balanced, sideFused_pixel_error


def train_generator(data):
    while True:
        batch_ids = np.random.choice(data.train_ids, data.batch_size)
        imgs, lbls = data.get_batches(batch_ids)
        yield (imgs, [lbls, lbls, lbls, lbls, lbls, lbls])


if __name__ == '__main__':
    # 注：训练需在tf2.0.0版本下进行
    # environment
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    K.set_image_data_format('channels_last')
    K.image_data_format()

    # prepare data
    with open('cfg.yml') as file:
        cfg = yaml.load(file)
    dataParser = DataParser(cfg)

    hed = HED(tf.keras.Input((cfg['height'], cfg['width'], cfg['channel'])), cfg['weight_decay_ratio'])
    hed.hed_cnn()

    # call backs
    model_save = callbacks.ModelCheckpoint(filepath=cfg['model_weights_path'], verbose=1, save_best_only=True)
    csv_logger = callbacks.CSVLogger(os.path.join(cfg['log_dir'], 'train_log.csv'), append=True, separator=';')
    tensor_board = callbacks.TensorBoard(log_dir=cfg['log_dir'], histogram_freq=0,
                                         write_graph=False, write_images=False)

    # training
    train_history = hed.model.fit_generator(
                        train_generator(dataParser),
                        steps_per_epoch=dataParser.steps_per_epoch,  # batch size
                        epochs=cfg['max_epochs'],
                        validation_data=None,
                        validation_steps=None,
                        callbacks=[model_save, csv_logger, tensor_board])

    hed.model.save(cfg['model_weights_path'] + 'model.h5')

    # print(train_history)