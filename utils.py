from __future__ import division
import numpy as np

def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    在两种坐标格式之间转换轴对齐2D框的坐标。
     下面有3中模式可以互相转换：
        1) (xmin, xmax, ymin, ymax) - 'minmax' 格式
        2) (xmin, ymin, xmax, ymax) - 'corners' 格式
        2) (cx, cy, w, h) - 'centroids' 格式
    Arguments:
        tensor (array): 一个Numpy nD数组，其中包含要在最后一个轴上的某个位置转换的四个连续坐标。
        start_index (int): 张量的最后一个轴上的第一个坐标的索引。
        conversion (str, optional): 转换方式。
        border_pixels (str, optional): 如何处理边界框的边框像素。
    Returns:
        一个Numpy nD数组，输入张量的副本，其中转换后的坐标代替原始坐标，而原始张量的未更改元素在其他位置。
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1

#===========================包围框工具===============================
def intersection_area_(boxes1, boxes2,  mode='outer_product'):
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    if mode == 'outer_product':

        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))

        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

        side_lengths = np.maximum(0, max_xy - min_xy)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

        side_lengths = np.maximum(0, max_xy - min_xy)

        return side_lengths[:, 0] * side_lengths[:, 1]

def iou(boxes1, boxes2, mode='outer_product', border_pixels='half'):
    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
    boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')

    intersection_areas = intersection_area_(boxes1, boxes2, mode=mode)

    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    if mode == 'outer_product':

        boxes1_areas = np.tile(
            np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d), axis=1),
            reps=(1, n))
        boxes2_areas = np.tile(
            np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d), axis=0),
            reps=(m, 1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas

#==============================真值匹配工具============================
def match_bipartite_greedy(weight_matrix):
    weight_matrix = np.copy(weight_matrix)
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes))

    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    for _ in range(num_ground_truth_boxes):
        anchor_indices = np.argmax(weight_matrix, axis=1) # 选取最大的值索引
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps)
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index

        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0

    return matches

def match_multi(weight_matrix, threshold):
    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes))

    ground_truth_indices = np.argmax(weight_matrix, axis=0)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]

    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met