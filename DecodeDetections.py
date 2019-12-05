"""
对预测结果的一个解码
输入:
    3维张量`(batch_size, n_boxes, n_classes + 12)`.
        batch_size:每次训练的数据量大小；
        n_boxes:预测的包围框的数目
        n_classes + 4 + 4 + 4:classes是独热编码。
输出:
    3维张量`(batch_size, top_k, 6)`.
        batch_size：每次训练的数据量大小；
        top_k：预测的包围框的数目
        6：[class_id, confidence, xmin, ymin, xmax, ymax]
"""

from __future__ import division
import tensorflow as tf
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine import Layer

class DecodeDetections(Layer):
    '''
    一个用于解析SSD预测输出的Keras层
    输入:
        3维张量`(batch_size, n_boxes, n_classes + 12)`.
    输出:
        3维张量`(batch_size, top_k, 6)`.
    '''

    def __init__(self,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=200,
                 nms_max_output_size=400,
                 normalize_coords=True,
                 img_height=None,
                 img_width=None,
                 **kwargs):
        '''
        参数:
            confidence_thresh (float, optional): 一个范围在[0,1)浮点型,在特定正类别中的最小分类置信度，
            以便考虑在各个类别的非最大抑制阶段。较低的值将导致选择过程的大部分由非最大抑制阶段完成，
            而较大的值将导致选择过程的较大部分发生在置信度阈值阶段。
            iou_threshold (float, optional): 一个范围在[0,1]浮点型. Jaccard相似度大于“ iou_threshold”且具有局部最大框
            的所有框将从给定类的预测集中删除，其中“最大”指框得分。
            top_k (int, optional): 在非最大抑制阶段之后，每个批次项目将保留的最高评分预测数。
            nms_max_output_size (int, optional): 执行非最大抑制后将剩余的最大预测数。
            coords (str, optional): 模型输出的包围框坐标格式。
            normalize_coords (bool, optional): 如果模型输出相对坐标（即[0,1]中的坐标），并且您希望将这些相对坐标
            转换回绝对坐标，则设置为“ True”。 否则为False。如果模型已经输出了绝对坐标，请不要将其设置为“ True”，
            否则会导致坐标不正确。 如果设置为True，则需要img_height和img_width。
            img_height (int, optional): 输入图像的高度。 仅当normalize_coords是True时才需要。
            img_width (int, optional): 输入图像的宽度。 仅当normalize_coords是True时才需要。
        '''
        # 需要对成员进行配置。
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.nms_max_output_size = nms_max_output_size

        # TensorFlow下需要的配置。
        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

        super(DecodeDetections, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetections, self).build(input_shape)

    def call(self, y_pred, mask=None):
        '''
        Returns:
            3维的张量 `(batch_size, top_k, 6)`. 第二个轴是零填充的，以便始终对每个批次项目产生“ top_k”预测。 最后一个轴以
            `[class_id, confidence, xmin, ymin, xmax, ymax]`格式包含每个预测框的坐标。
        '''

        #####################################################################################
        # 1. 将框坐标从预测的锚框偏移量转换为预测的绝对坐标
        #####################################################################################
        # 将锚框偏移量转换为图像偏移量。
        cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[..., -8] # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[..., -7] # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        w = tf.exp(y_pred[..., -10] * y_pred[..., -2]) * y_pred[..., -6] # w = exp(w_pred * variance_w) * w_anchor
        h = tf.exp(y_pred[..., -9] * y_pred[..., -1]) * y_pred[..., -5] # h = exp(h_pred * variance_h) * h_anchor

        # 转换 'centroids' 维 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        # 如果模型预测相对于图像尺寸的框坐标，并且应该将其转换回绝对坐标，请执行此操作。
        def normalized_coords():
            xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1
        def non_normalized_coords():
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(self.tf_normalize_coords, normalized_coords, non_normalized_coords)

        # 将one-hot类置信度与转换后的框坐标连接起来，以形成解码后的预测张量。
        y_pred = tf.concat(values=[y_pred[..., :-12], xmin, ymin, xmax, ymax], axis=-1)

        #####################################################################################
        # 2. 执行置信度阈值，每类非最大抑制和top-k过滤。
        #####################################################################################

        n_classes = y_pred.shape[2] - 4

        # 创建一个函数，该函数可以过滤给定批次项目的预测。 具体来说，它执行:
        # - confidence thresholding
        # - non-maximum suppression (NMS)
        # - top-k filtering
        def filter_predictions(batch_item):

            # 创建一个函数来过滤单个类的预测。
            def filter_single_class(index):
                # 从一个张量(n_boxes, n_classes + 4 coordinates) 提取
                # 一个张量 (n_boxes, 1 + 4 coordinates) 它包含一类的置信值，通过`index`确定。
                confidences = tf.expand_dims(batch_item[..., index], axis=-1)
                class_id = tf.fill(dims=tf.shape(confidences), value=tf.to_float(index))
                box_coordinates = batch_item[..., -4:]

                single_class = tf.concat([class_id, confidences, box_coordinates], axis=-1)

                # 通过`index`将置信值对应到类
                threshold_met = single_class[:, 1] > self.tf_confidence_thresh
                single_class = tf.boolean_mask(tensor=single_class,
                                               mask=threshold_met)

                # 如果有任何框达到阈值，请执行NMS。
                def perform_nms():
                    scores = single_class[..., 1]

                    # `tf.image.non_max_suppression()` 需要包围框的格式 `(ymin, xmin, ymax, xmax)`.
                    xmin = tf.expand_dims(single_class[..., -4], axis=-1)
                    ymin = tf.expand_dims(single_class[..., -3], axis=-1)
                    xmax = tf.expand_dims(single_class[..., -2], axis=-1)
                    ymax = tf.expand_dims(single_class[..., -1], axis=-1)
                    boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                    maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                                  scores=scores,
                                                                  max_output_size=self.tf_nms_max_output_size,
                                                                  iou_threshold=self.iou_threshold,
                                                                  name='non_maximum_suppresion')
                    maxima = tf.gather(params=single_class,
                                       indices=maxima_indices,
                                       axis=0)
                    return maxima

                def no_confident_predictions():
                    return tf.constant(value=0.0, shape=(1, 6))

                single_class_nms = tf.cond(tf.equal(tf.size(single_class), 0), no_confident_predictions, perform_nms)

                # 确保“ single_class”与“ self.nms_max_output_size”元素完全相同。
                padded_single_class = tf.pad(tensor=single_class_nms,
                                             paddings=[[0, self.tf_nms_max_output_size - tf.shape(single_class_nms)[0]], [0, 0]],
                                             mode='CONSTANT',
                                             constant_values=0.0)

                return padded_single_class

            # 在所有类索引上迭代`filter_single_class（）`。
            filtered_single_classes = tf.map_fn(fn=lambda i: filter_single_class(i),
                                                elems=tf.range(1, n_classes),
                                                dtype=tf.float32,
                                                parallel_iterations=128,
                                                back_prop=False,
                                                swap_memory=False,
                                                infer_shape=True,
                                                name='loop_over_classes')

            # 将所有单个类的过滤结果连接到一个张量。
            filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1, 6))

            # 对这个批处理执行top-k过滤，或者填充，以防万一此时剩下的`self.top_k`框过少。
            #无论哪种方式，都产生一个长度为 `self.top_k` 的张量。当我们返回整个批次的最终结果张量时，
            # 所有批次项目都必须具有相同数量的预测框，以使张量尺寸均匀。如果在上述过滤过程后剩下的预测少于self.top_k，
            # 则我们用零填充缺失的预测作为虚拟条目。
            def top_k():
                return tf.gather(params=filtered_predictions,
                                 indices=tf.nn.top_k(filtered_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)
            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=filtered_predictions,
                                            paddings=[[0, self.tf_top_k - tf.shape(filtered_predictions)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], self.tf_top_k), top_k, pad_and_top_k)

            return top_k_boxes

        # 在所有批处理项目上迭代`filter_predictions（）`。
        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return (batch_size, self.tf_top_k, 6) # 最后一维表示: (class_ID, confidence, 4 box coordinates)