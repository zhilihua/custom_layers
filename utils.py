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