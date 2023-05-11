from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf

class RunTensorRT():
    def __init__(self, model_path):
        self.model_path = model_path

    def get_model(self):
        saved_model_loaded = tf.saved_model.load(
            self.model_path, tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.tensor_model = graph_func

    def run_pred(self, data, dtype = tf.float32):
        pred = self.tensor_model(tf.convert_to_tensor((data), dtype = dtype))
        return pred

