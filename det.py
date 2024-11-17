import torch
import sys

import pdb
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, create_dataloader
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,xywh2xyxy)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox 
import numpy as np
from pathlib import Path
from utils.metrics import ConfusionMatrix, ap_per_class
from collections import Counter

def iou(box1, box2):
    inter_x = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_y = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if inter_x < 0 or inter_y < 0 else inter_x * inter_y
    union = (box2[2]-box2[0]) * (box2[3]-box2[1]) + (box1[2]-box1[0]) * (box1[3]-box1[1]) - inter
    iou = inter / union
    return iou


class predict:
    def __init__(self):
        self.weights = "./yolov5l.pt"  # model.pt path(s)
        self.data = "data/coco128.yaml"  # dataset.yaml path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.half = False
        self.dnn = True  # use OpenCV DNN for ONNX inference
 
        # Load model
        self.device = torch.device('cuda:0')
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        # Half
        self.half = False  # FP16 supported on limited backends with CUDA
 
    # 读取标注文件
    def read_labels(self, label_file):
        boxes = []
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])  # 类别ID
                x_center, y_center, w, h = map(float, parts[1:])
                # 转换为框的左上角和右下角坐标（归一化坐标）# Kun(2024/1116):相对于整图、
                x1 = (x_center - w / 2)
                y1 = (y_center - h / 2)
                x2 = (x_center + w / 2)
                y2 = (y_center + h / 2)
                boxes.append((class_id, x1, y1, x2, y2))
        return boxes

    def voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec=np.concatenate(([0.],rec,[1.]))
        mpre=np.concatenate(([0.],prec,[0.]))

        # 为求得曲线面积，对 Precision-Recall plot进行“平滑”处理
        for i in range(mpre.size -1, 0, -1):
            mpre[i-1]=np.maximum(mpre[i-1],mpre[i])

        # 找到 recall值变化的位置
        i=np.where(mrec[1:]!=mrec[:-1])[0]

        # recall值乘以相对应的 precision 值相加得到面积
        ap = np.sum((mrec[i+1]-mrec[i])*mpre[i+1])
        return ap    

    def calculate_map_at_iou(self, pred, true_boxes, iou_threshold):
        ap = []
        # 统计每个第一个值的出现次数
        count = Counter(item[0] for item in true_boxes)

        # 转换为字典格式并打印
        true_count = dict(count)    # {72: 1, 69: 1, 68: 1, 60: 1, 56: 1, 45: 6, 41: 10, 40: 9, 39: 10}
        
        tp = {int(row[-1]): [] for row in pred}
        fp = {int(row[-1]): [] for row in pred}
        for pred_xmin, pred_ymin, pred_xmax, pred_ymax, conf, pred_class_id in pred:
            flag = 0
            for true_class_id, true_xmin, true_ymin, true_xmax, true_ymax in true_boxes:
                if pred_class_id == true_class_id and iou((pred_xmin, pred_ymin, pred_xmax, pred_ymax), (true_xmin, true_ymin, true_xmax, true_ymax)) > iou_threshold:
                    flag = 1
                    tp[pred_class_id].append(1)
                    fp[pred_class_id].append(0)
                    break
            if flag == 0:
                tp[pred_class_id].append(0)
                fp[pred_class_id].append(1)        
        ap = []
        for class_id in tp.keys():         
            tp_numpy = np.array(tp[class_id])
            fp_numpy = np.array(fp[class_id])
            tp_numpy = np.cumsum(tp_numpy)
            fp_numpy = np.cumsum(fp_numpy)
            current = np.maximum(tp_numpy + fp_numpy, np.finfo(np.float64).eps)
            prec = tp_numpy / current
            rec = (tp_numpy / true_count[class_id]) if class_id in true_count.keys() else np.array([0]*(len(prec)))
            ap.append(self.voc_ap(rec, prec))
        return np.sum(ap) / len(true_count)  #考虑到有些检测出的类在label中都没出现过，分母应为len(true_count)




    def detect_image(self, image_path, label_path):
        true_boxes = self.read_labels(label_path)
        self.model.eval()
        im0s = cv2.imread(image_path)
        im = letterbox(im0s, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im0 = im0s.copy()
        height, weigh, channel = im0.shape[0], im0.shape[1], im0.shape[2]
        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        # pdb.set_trace()
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # 转换为[[xmin,ymin,xmax,ymax,conf,cls]]
            det_list = [[box[0].item()/weigh, box[1].item()/height,
                            box[2].item()/weigh, box[3].item()/height, box[4].item()
                            , int(box[5].item())] for box in det]

        
        true_boxes = sorted(true_boxes, key = lambda x : x[0], reverse=True)
        # 按照类和置信度排序
        det_list = sorted(det_list, key = lambda x: (x[-1], x[-2]), reverse=True)
        mAP_50 = self.calculate_map_at_iou(det_list, true_boxes, 0.75)
        return mAP_50



        
if __name__ == "__main__":
    # print(predict().detect_image("datatest/000000000009.jpg",label_path="datasetTxt/000000000009.txt"))
    # print(predict().detect_image("datatest/000000000089.jpg",label_path="datasetTxt/000000000089.txt"))
    # print(predict().detect_image("datatest/000000000110.jpg",label_path="datasetTxt/000000000110.txt"))
    # print(predict().detect_image("datatest/000000000133.jpg",label_path="datasetTxt/000000000133.txt"))
    # print(predict().detect_image("datatest/000000000144.jpg",label_path="datasetTxt/000000000144.txt"))
    print(predict().detect_image("datatest/000000000164.jpg",label_path="datasetTxt/000000000164.txt"))
    # print(predict().detect_image("datatest/000000000009.jpg",label_path="datasetTxt/000000000009.txt"))
