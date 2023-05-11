import tensorrt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
import tensorflow as tf
import numpy as np

class CreateEngine():
    def __init__(self, model_path, data_gen, shape, batch_size):
        self.model_path = model_path
        self.data_gen = data_gen
        self.shape = shape
        self.batch_size = batch_size

    def load_model(self):
        converter = trt.TrtGraphConverterV2(
           input_saved_model_dir = self.model_path,
           precision_mode=trt.TrtPrecisionMode.FP32
        )
        self.trt_model = converter.convert()
        self.converter = converter

    def create_dataset(self):
        imgs = np.zeros(shape = self.shape)
        def read_and_batch_images(img):
            img = tf.io.read_file(img)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, size = (self.shape[1], self.shape[2]))
            img = img / 255
            return img
        count = 0
        for x in self.data_gen:
            i = read_and_batch_images(x)
            imgs[count] = i
            count += 1
        imgs = imgs.astype(np.float32)
        self.imgs = imgs
    
    def build_model(self):
        def input_fn():
            x = self.imgs[:self.batch_size, :]
            yield [x]
        self.converter.build(input_fn = input_fn)
    
    def save_model(self, rt_model_path):
        self.converter.save(rt_model_path)
    