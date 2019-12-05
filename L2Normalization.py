"""
自定义的标准化
"""
from __future__ import division

import tensorflow as tf
keras = tf.keras

K = keras.backend
InputSpec = keras.layers.InputSpec
class L2Normalization(keras.layers.Layer):
    '''
    在输入张量上进行可学习参数的L2标准化
    参数:
        gamma_init (int): 初始化比例参数。在SSD论文中默认值为20.
    Input shape:
        4维的形状 `(batch, height, width, channels)` .
    Returns:
        标准化的张量，与输入尺寸一样。
    '''

    def __init__(self, gamma_init=20, **kwargs):
        self.axis = 3
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.gamma = self.add_weight(name='{}_gamma'.format(self.name),
                                    shape=(input_shape[self.axis],),
                                    initializer=keras.initializers.Constant(value=self.gamma_init),
                                    trainable=True)

        super(L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        return output * self.gamma