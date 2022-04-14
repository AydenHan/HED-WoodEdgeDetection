import numpy as np
import cv2 as cv
import yaml
import os


class DataParser(object):
    def __init__(self, cfg):
        self.width = cfg['width']
        self.height = cfg['height']
        self.channel = cfg['channel']
        self.batch_size = cfg['batch_size']
        self.mean = cfg['mean']

        with open(cfg['train_list']) as f:
            file_list = f.readlines()
        file_list = [file.strip().split(' ') for file in file_list]
        self.samples = [(os.path.join(cfg['train_dir'], file[0]), os.path.join(cfg['train_dir'], file[1]))
                        for file in file_list]
        samples_num = len(self.samples)
        ids = range(samples_num)
        np.random.shuffle(list(ids))

        self.train_ids = ids[:samples_num - samples_num % self.batch_size]
        self.validation_ids = ids[samples_num - samples_num % self.batch_size:]
        self.steps_per_epoch = len(self.train_ids) / self.batch_size

        # self.use_validation = len(self.validation_ids) != 0
        # self.validation_steps = len(self.validation_ids)/(batch_size*2)

    def get_batches(self, batch_ids):
        images = []
        labels = []

        for index, id in enumerate(batch_ids):
            image = cv.imread(self.samples[id][0])
            label = cv.imread(self.samples[id][1], cv.IMREAD_GRAYSCALE)
            image = np.array(image, dtype=np.float32)
            # image -= self.mean
            image[..., 0] -= self.mean[0]
            image[..., 1] -= self.mean[1]
            image[..., 2] -= self.mean[2]
            label = np.array(label / 255, dtype=np.float32)

            images.append(image)
            labels.append(label)

        images = np.asarray(images)
        labels = np.asarray(labels)

        return images, labels

    def get_single_test(self, batch_ids):
        images = []

        for index, id in enumerate(batch_ids):
            image = cv.imread(self.samples[id][0])
            print(self.samples[id][0])
            image = np.array(image, dtype=np.float32)
            # image -= self.mean
            image[..., 0] -= self.mean[0]
            image[..., 1] -= self.mean[1]
            image[..., 2] -= self.mean[2]

            images.append(image)

        images = np.asarray(images)

        return images



