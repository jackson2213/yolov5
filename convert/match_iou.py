import numpy as np
from scipy.optimize import linear_sum_assignment

threshold=0.7

def calculate_iou(box1, box2):
    """
    计算两个矩形框的IoU（交并比）。

    参数:
        box1 (tuple): 第一个矩形框，格式为(x1, y1, x2, y2)。
        box2 (tuple): 第二个矩形框，格式为(x1, y1, x2, y2)。

    返回:
        float: IoU值。
    """
    if  len(box1) ==0 or len(box2) ==0:
        return 0
    try:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # 计算相交区域面积
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # 计算两个矩形框的面积
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # 计算并集面积
        union_area = box1_area + box2_area - intersection_area

        # 计算IoU
        iou = intersection_area / union_area

        return iou
    except Exception as e:
        print(f"box1: {box1}, box2: {box2}, error: {e}")
        return 0


def match_boxes_with_xyl(prev_boxes, current_boxes, threshold=threshold):
    """
    尝试匹配前一帧和当前帧的目标框。

    参数:
        prev_boxes (list of tuples): 上一帧的目标框列表。
        current_boxes (list of tuples): 当前帧的目标框列表。
        threshold (float): IoU阈值，超过此阈值则认为匹配。

    返回:
        tuple: 包含三个字典的元组，第一个字典表示匹配结果，
               键为前一帧的目标框索引，值为当前帧的目标框索引；
               第二个字典表示当前帧中新出现的目标框索引；
               第三个字典表示前一帧中未能匹配的目标框索引。
    """
    # 检查输入的有效性
    # 检查输入的有效性
    if not prev_boxes and not current_boxes:
        return {}, {}, {}
    elif not prev_boxes:
        return {}, {idx: True for idx in range(len(current_boxes))}, {}
    elif not current_boxes:
        return {}, {}, {idx: True for idx in range(len(prev_boxes))}

    matches = {}
    new_targets = {}
    disappeared_targets = {}

    num_prev_boxes = len(prev_boxes)
    num_current_boxes = len(current_boxes)

    # 构建成本矩阵
    cost_matrix = np.zeros((num_prev_boxes, num_current_boxes))
    for i, prev_box in enumerate(prev_boxes):
        for j, current_box in enumerate(current_boxes):
            iou = calculate_iou(prev_box, current_box)
            cost_matrix[i, j] = -iou  # 负数表示最大化IoU

    # 使用匈牙利算法求解
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    for i, j in zip(row_ind, col_ind):
        if -cost_matrix[i, j] > threshold:
            matches[i] = j

    # 未被匹配的目标框被视为新出现的目标
    new_targets = set(range(num_current_boxes)) - set(col_ind)
    new_targets = {j: True for j in new_targets}

    # 未被匹配的前一帧目标框被视为消失的目标
    disappeared_targets = set(range(num_prev_boxes)) - set(row_ind)
    disappeared_targets = {i: True for i in disappeared_targets}

    return matches, new_targets, disappeared_targets

def match_boxes_with_iou(prev_boxes, current_boxes, threshold=threshold):
    """
    尝试匹配前一帧和当前帧的目标框。

    参数:
        prev_boxes (list of tuples): 上一帧的目标框列表。
        current_boxes (list of tuples): 当前帧的目标框列表。
        threshold (float): IoU阈值，超过此阈值则认为匹配。

    返回:
        tuple: 包含三个字典的元组，第一个字典表示匹配结果，
               键为前一帧的目标框索引，值为当前帧的目标框索引；
               第二个字典表示当前帧中新出现的目标框索引；
               第三个字典表示前一帧中未能匹配的目标框索引。
    """
    # 检查输入的有效性
    if not prev_boxes and not current_boxes:
        return {}, {}, {}
    elif not prev_boxes:
        return {}, {idx: True for idx in range(len(current_boxes))}, {}
    elif not current_boxes:
        return {}, {}, {idx: True for idx in range(len(prev_boxes))}

    matches = {}
    new_targets = {}
    disappeared_targets = {}

    # 用于标记当前帧中的目标框是否已被匹配
    matched_current_boxes = [False] * len(current_boxes)

    # 用于标记前一帧中的目标框是否已被匹配
    matched_prev_boxes = [False] * len(prev_boxes)

    for i, prev_box in enumerate(prev_boxes):
        best_iou = 0
        best_j = -1
        for j, current_box in enumerate(current_boxes):
            if matched_current_boxes[j]:
                continue
            iou = calculate_iou(prev_box, current_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou > threshold:
            matches[i] = best_j
            matched_current_boxes[best_j] = True
            matched_prev_boxes[i] = True

    # 未被匹配的目标框被视为新出现的目标
    for j, is_matched in enumerate(matched_current_boxes):
        if not is_matched:
            new_targets[j] = True

    # 未被匹配的前一帧目标框被视为消失的目标
    for i, is_matched in enumerate(matched_prev_boxes):
        if not is_matched:
            disappeared_targets[i] = True

    return matches, new_targets, disappeared_targets


# 示例用法
# prev_detections = [
#     ('person', (100, 100, 200, 200)),
#     ('car', (300, 300, 400, 400)),
#     ('bike', (500, 500, 600, 600))
# ]
#
# current_detections = [
#     ('person', (105, 105, 205, 205)),
#     ('car', (305, 305, 405, 405)),
#     ('new_person', (500, 500, 600, 600))
# ]
#
# # 提取边界框
# prev_boxes = [detection[1] for detection in prev_detections]
# current_boxes = [detection[1] for detection in current_detections]
#
# # 匹配目标框
# matches, new_targets, disappeared_targets = match_boxes(prev_boxes, current_boxes, threshold=0.5)
#
# # 输出匹配结果
# print("Matches:", matches)
# print("New Targets:", new_targets)
# print("Disappeared Targets:", disappeared_targets)