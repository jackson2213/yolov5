import os
import cv2
import sys
import argparse

# add path
import numpy as np
import cv2
from match_iou import match_boxes_with_iou, match_boxes_with_xyl
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shutil

# Model from https://github.com/airockchip/rknn_model_zoo
ONNX_MODEL = 'best.onnx'
RKNN_MODEL = 'yolov5s_relu.rknn'
IMG_PATH = './bus.jpg'
DATASET = './dataset.txt'

QUANTIZE_ON = True

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES =  ['Garbage','Person','hold garbage','trash bin']



def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]

    box_xy = input[..., :2]*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(input[..., 2:4]*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        if cl==1:
            continue
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 1)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def is_cls_coordinate_in_detect_area(x,y):
    # 05 (220,150), (250, 492), (550, 495),(450, 160)
    # 07 (40,200), (80, 380), (600, 350),(550, 200)
    polygon_coordinates = [(220,150), (250, 492), (550, 495),(450, 160)]
    polygon = Polygon(polygon_coordinates)
    point = Point(x, y)
    if polygon.contains(point):
        return True
    return False


def is_point_in_polygon(point, polygon):
    """
    检查一个点是否位于指定的多边形区域内。

    参数:
        point (tuple): 待测点坐标，格式为(cx, cy)。
        polygon (list of tuples): 多边形顶点坐标列表，格式为[(x1, y1), (x2, y2), ...]。

    返回:
        bool: 如果点在多边形内返回True，否则返回False。
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def parse_write(f,boxes,ori_matches,index1,index2,alg):
    if alg== "iou":
       matches, new_targets, disappeared_targets = match_boxes_with_iou(boxes[index1], boxes[index2])
    else:
       matches, new_targets, disappeared_targets = match_boxes_with_xyl(boxes[index1], boxes[index2])
    # 输出匹配结果
    result = ""
    if not ori_matches:
        if len(new_targets) > 0:
            f.write(f"detect_result: No new targets found\n")
            f.write(f"detect_result: ADD \n")
            result = "ADD"
        else:
            f.write(f"detect_result: Disappeared targets: {disappeared_targets}\n")
            f.write(f"detect_result: DROP \n")
            result = "DROP"
    else:
        if ori_matches == matches:
            if len(new_targets) > 0:
                f.write(f"detect_result: New targets: {new_targets}\n")
                f.write(f"detect_result: ADD \n")
                result = "ADD"
            else:
                f.write(f"detect_result: Disappeared targets: {disappeared_targets}\n")
                f.write(f"detect_result: DROP \n")
                result = "DROP"
        else:
            if len(new_targets) > 0 and len(disappeared_targets) > 0:
                f.write(f"detect_result: New targets: {new_targets},Disappeared targets: {disappeared_targets}\n")
                f.write(f"detect_result: ADD \n")
                result = "ADD"
            elif len(new_targets) > 0:
                f.write(f"detect_result: New targets: {new_targets}\n")
                f.write(f"detect_result: ADD \n")
                result = "ADD"
            elif len(disappeared_targets) > 0:
                f.write(f"detect_result: Disappeared targets: {disappeared_targets}\n")
                f.write(f"detect_result: DROP \n")
                result = "DROP"
            else:
                f.write(f"detect_result: ori_matches:{ori_matches},New targets: {new_targets}\n")
                f.write(f"detect_result: ADD \n")
                result = "ADD"

    f.write('**' * 20 + "\n")
    f.write(f"Matches:ori:{ori_matches},latest:{matches}\n")
    f.write(f"New Targets:{new_targets}\n")
    f.write(f"Disappeared Targets:{disappeared_targets}\n")
    print("\n\n")
    print("*" * 20)
    return result




def detect_mp4(model,path,save_result=False):
    video = cv2.VideoCapture(path)
    irregular_rectangles = {"000003": [(100, 200), (80, 492), (500, 495), (500, 180)],
                           "000004": [(200, 250), (256, 492), (500, 500), (400, 250)],
                           "000005": [(220, 150), (250, 492), (550, 495), (450, 180)],
                           "000006": [(140, 250), (160, 492), (500, 495), (300, 280)],
                           "000007": [(40,200), (80, 380), (600, 310),(550, 200)]
                           }
    sn=path.split("\\")[-1].split(".")[0].split("_")[-2]
    if not video.isOpened():
        print("无法打开视频文件")
        exit()
    count = 0
    result = []
    # 定义编码器对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 也可以使用其他编码器，例如 'mp4v'
    # 获取视频的一些属性
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    fps = video.get(cv2.CAP_PROP_FPS)
    # 创建一个VideoWriter对象来输出视频
    out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

    while video.isOpened():
        count = count + 1
        # 逐帧读取视频
        ret, frame = video.read()
        if ret == False:
            break
        img_name = "result/test_" + str(count) + ".jpg"
        # Set inputs
        img, ratio, (dw, dh) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))

        # Inference
        print('--> Running model')
        outputs = model.run(inputs=[img])

        input0_data = outputs[0].reshape([3, -1] + list(outputs[0].shape[-2:]))
        input1_data = outputs[1].reshape([3, -1] + list(outputs[1].shape[-2:]))
        input2_data = outputs[2].reshape([3, -1] + list(outputs[2].shape[-2:]))
        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
        temp_data = []
        boxes, classes, scores = yolov5_post_process(input_data)
        # 对每一帧进行处理，这里只是一个简单的复制

        if boxes is not None:
            for cl, box, sc in zip(classes, boxes, scores):
                if cl == 0 and sc > 0.3:
                    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    is_in_box = is_point_in_polygon((center_x, center_y), irregular_rectangles[sn])
                    print(f"center_x:{center_x},center_y:{center_y},is_in_box:{is_in_box}")
                    if is_in_box:
                        temp_data.append(box)
            result.append(temp_data)
            draw(img, boxes, scores, classes)
            if save_result:
                draw(img, boxes, scores, classes)
                cv2.imwrite(img_name, img)
        out.write(img)

    video.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    for alg in ["iou", "xyl"]:
        file_name = path.split('/')[-1].split('.')[0]
        result_path = "result_" + alg
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        with open(f'{os.path.join(result_path, file_name)}_boxes.txt', 'w') as f:
            for item in result:
                f.write(f"{item}\n")

        status = write_to_txt(file_name, result, alg)
        shutil.copy(f"output_video.avi", f"{result_path}/{file_name}_{status}.avi")
    os.remove("output_video.avi")








def write_to_txt(file_name,result,alg):
    result_path = "result_" + alg
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(f'{os.path.join(result_path, file_name)}.txt', 'w') as f:
        if len(result[0])>0 and len(result[1])>0:
            if alg=="iou":
                matches, new_targets, disappeared_targ = match_boxes_with_iou(result[0], result[1])
            else:
                matches, new_targets, disappeared_targ = match_boxes_with_xyl(result[0], result[1])
            if len(matches)==len(result[0]):
               f.write("found targets in detected area\n")
               return parse_write(f,result,matches,0,-1,alg)
            else:
                f.write(f"Not found targets in detected area:  matches:{matches},new_targets:{new_targets},disappeared_targets:{disappeared_targ}\n")
                return parse_write(f, result, {}, 0, -1, alg)

        else:
            f.write("No targets in detected area\n")
            return parse_write(f, result, {}, 0, -1, alg)


# # 示例用法
# center_point = (150, 150)  # 中心点坐标
# irregular_rectangle = [(100, 100), (200, 100), (200, 200), (150, 250), (100, 200)]  # 不规则矩形的顶点坐标
#
# # 检查中心点是否在不规则矩形内
# is_in_irregular_rectangle = is_point_in_polygon(center_point, irregular_rectangle)

#print(f"Center point ({center_point}) is in the irregular rectangle: {is_in_irregular_rectangle}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')

    # coco val folder: '../../../datasets/COCO//val2017'
    parser.add_argument('--data', type=str, default='./data', help='img folder path')
    parser.add_argument('--save_result', type=bool, default=False, help='img folder path')


    args = parser.parse_args()
    model, platform = setup_model(args)
    video_list=[]
    if os.path.isfile(args.data) and args.data.endswith('.mp4'):
        video_list.append(args.data)
    elif os.path.isdir(args.data):
        for root, dirs, files in os.walk(args.data):
            for file in files:
                if file.endswith('.mp4'):
                    video_list.append(os.path.join(root, file))
    else:
        print("输入文件路径有误")

    for path in video_list:
        detect_mp4(model, path, args.save_result)




