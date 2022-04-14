from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from hed import cross_entropy_balanced, sideFused_pixel_error


if __name__ == "__main__":
    # 注：转化需在tf2.3.0版本下进行
    # 将h5模型转化为tflite模型方法
    modelparh = './data/weights/model.h5'
    model = tf.keras.models.load_model(modelparh, custom_objects = {'cross_entropy_balanced': cross_entropy_balanced, 'sideFused_pixel_error': sideFused_pixel_error})

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()
    savepath = './data/weights/model.tflite'
    open(savepath, "wb").write(tflite_model)
