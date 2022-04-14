from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Concatenate, Activation
from tensorflow.keras import backend as K
import tensorflow as tf
import yaml


def cross_entropy_balanced(y_true, y_pred):
    _epsilon = tf.convert_to_tensor(K.epsilon())
    if _epsilon.dtype != y_pred.dtype.base_dtype:
        _epsilon = tf.cast(_epsilon, y_pred.dtype.base_dtype)

    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def sideFused_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


class HED(object):
    def __init__(self, img, decay_ratio):
        self.input = img
        self.decay_ratio = decay_ratio
        self.model = None

    def hed_cnn(self):
        bn1, out1 = self.hed_layer(self.input, 64, iteration=2, name='block1', dilation_rate=[(4, 4), (1, 1)])
        pool1 = MaxPool2D((2, 2), (2, 2), 'same', name='pool1')(out1)

        bn2, out2 = self.hed_layer(pool1, 128, iteration=2, name='block2')
        pool2 = MaxPool2D((2, 2), (2, 2), 'same', name='pool2')(out2)

        bn3, out3 = self.hed_layer(pool2, 256, iteration=3, name='block3')
        pool3 = MaxPool2D((2, 2), (2, 2), 'same', name='pool3')(out3)

        bn4, out4 = self.hed_layer(pool3, 512, iteration=3, name='block4')
        pool4 = MaxPool2D((2, 2), (2, 2), 'same', name='pool4')(out4)

        bn5, out5 = self.hed_layer(pool4, 512, iteration=3, name='block5')

        side1 = self.hed_side(bn1, 1, 'side1', False)
        side2 = self.hed_side(bn2, 2, 'side2')
        side3 = self.hed_side(bn3, 4, 'side3')
        side4 = self.hed_side(bn4, 8, 'side4')
        side5 = self.hed_side(bn5, 16, 'side5')

        sideFused = Concatenate(axis=-1)([side1, side2, side3, side4, side5])
        sideFused = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None, name='sideFused-conv')(sideFused)
        sideFused = Activation('sigmoid', name='sideFused')(sideFused)

        self.model = tf.keras.Model([self.input], [side1, side2, side3, side4, side5, sideFused])
        self.model.compile(loss={'side1': cross_entropy_balanced,
                                 'side2': cross_entropy_balanced,
                                 'side3': cross_entropy_balanced,
                                 'side4': cross_entropy_balanced,
                                 'side5': cross_entropy_balanced,
                                 'sideFused': cross_entropy_balanced},
                           metrics={'sideFused': sideFused_pixel_error},
                           optimizer='adam')
        return self.model

    def hed_layer(self, input_tensor, filters, iteration, name=None, dilation_rate=None):
        input = input_tensor
        if dilation_rate is None:
            dilation_rate = [(1, 1)]
        if len(dilation_rate) == 1:
            dilation_rate *= iteration

        regularizer = tf.keras.regularizers.l2(self.decay_ratio)

        for it in range(iteration):
            tp_dilation_rate = dilation_rate.pop(0)
            conv = Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu',
                          use_bias=True,
                          kernel_regularizer=regularizer,
                          dilation_rate=tp_dilation_rate,
                          name=name + '-conv{:d}'.format(it))(input)
            input = conv

        return input, input

    def hed_side(self, input_tensor, stride, name=None, deconv=True):
        side = Conv2D(1, (1, 1), (1, 1), 'same',
                      kernel_regularizer=tf.keras.regularizers.l2(self.decay_ratio),
                      activation=None,
                      name=name + '-conv')(input_tensor)
        if deconv:
            side = Conv2DTranspose(1, (2*stride, 2*stride), (stride, stride), 'same',
                                   kernel_regularizer=tf.keras.regularizers.l2(self.decay_ratio),
                                   activation=None,
                                   use_bias=False,
                                   name=name + '-convT')(side)

        side = Activation('sigmoid', name=name)(side)
        return side


if __name__ == '__main__':
    with open('cfg.yml') as file:
        cfg = yaml.load(file)
    hed = HED(tf.keras.Input((cfg['width'], cfg['height'], cfg['channel'])), cfg['weight_decay_ratio'])
    m = hed.hed_cnn()
    m.summary()

