"""
该方法是基于SSD生成的锚点，基于不同的生成策略进行相应的修改，
大部分的思路是相同的。
注意输入时一个4维向量，输出是一个5维向量。
"""

from __future__ import division
import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine import Layer

from utils import convert_coordinates

class AnchorBoxes(Layer):
    '''
    Input shape:
        4D `(batch, height, width, channels)` .
    Output shape:
        5D `(batch, height, width, n_boxes, 8)`.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):
        '''
        Arguments:
            img_height (int): 输入图片高度。
            img_width (int): 输入图片宽度。
            this_scale (float): [0, 1]之间的浮点数。生成锚点包围框时候用到。
            next_scale (float): [0, 1]之间的浮点数。生成锚点包围框时候用到。仅仅当`self.two_boxes_for_ar1 == True`.
            aspect_ratios (list, optional): 此层生成默认框的纵横比的列表。
            two_boxes_for_ar1 (bool, optional): 仅在`aspect_ratios`包含1时才相关.
            variances (list, optional): 一个拥有四个大于零的浮点值列表。每个坐标的锚框偏移将除以其各自的方差值。
            coords (str, optional): 包围框坐标转换方式。
            normalize_coords (bool, optional): 是否进行标准化。
        '''
        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords
        # 计算每一个中心的包围框数量。
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        根据输入张量的形状返回锚框张量。
        Arguments:
            x (张量): 4D `(batch, height, width, channels)` . 该层的输入必须是本地化预测器层的输出。
        '''
        #=====================================不同策略仅需修改下面这一部分代码=================================
        # 计算每一个宽高比下的宽和高。
        # 图像将根据`scale` 和 `aspect_ratios`并利用较短的边计算`w` and `h`。
        size = min(self.img_height, self.img_width)
        # 计算所有纵横比的框宽和高
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # 计算宽高比为1.的常规锚框。
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # 使用此比例尺值的几何平均值计算一个稍大的包围框。
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        #=====================================================================================================
        # 输入的shape，这是我们所必须的
        batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape

        # 获取step尺寸。
        step_height = self.this_steps
        step_width = self.this_steps
        # 获取offsets尺寸。
        offset_height = self.this_offsets
        offset_width = self.this_offsets
        # 现在我们有了偏移量和步长，计算锚点盒中心点的网格。
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        # 创建一个4D模板`(feature_map_height, feature_map_width, n_boxes, 4)`，这里最后一个维度包含`(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # 设置 cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # 设置 cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # 设置 w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # 设置 h

        # 转换 `(cx, cy, w, h)` 为 `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # 进行标准化，使所有值在[0, 1]。
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        if self.coords == 'centroids':
            # 转换 `(xmin, ymin, xmax, ymax)` 为 `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # 转换 `(xmin, ymin, xmax, ymax)` 为 `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        variances_tensor = np.zeros_like(boxes_tensor) # shape为 `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances
        # 现在 `boxes_tensor` 变为形状`(feature_map_height, feature_map_width, n_boxes, 8)`的张量。
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)