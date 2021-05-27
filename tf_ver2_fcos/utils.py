import cv2
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.
    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.
    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], 
                     boxes[:, 3], boxes[:, 2]], axis=-1)

def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, 
         boxes[..., 2:] - boxes[..., :2]], axis=-1)

def convert_to_corners(boxes):
    """Changes the box format to corner coordinates
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, 
         boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1)

def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes
    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

def visualize_detections(
    image, boxes, classes, scores, 
    figsize=(7, 7), linewidth=1, color=[0, 0, 1], 
    save_img_file="detection.jpg", show_text=True):
    """Visualize Detections"""
    tmp_img  = np.array(image, dtype=np.uint8)
    show_img = np.zeros_like(tmp_img)
    
    
    # Swap R and B channels for proper display. #
    show_img[:, :, 0] = tmp_img[:, :, 2]
    show_img[:, :, 1] = tmp_img[:, :, 1]
    show_img[:, :, 2] = tmp_img[:, :, 0]
    del tmp_img
    
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        
        # Convert to numpy. #
        x1 = max(int(x1.numpy()), 0)
        x2 = int(x2.numpy())
        y1 = max(int(y1.numpy()), 0)
        y2 = int(y2.numpy())
        
        cv2.rectangle(
            show_img, (x1,y1), (x2,y2), (255,0,0))
        
        if show_text:
            cv2.putText(
                show_img, text, (x1,y1), 
                cv2.FONT_HERSHEY_DUPLEX, 
                fontScale=0.5, color=(255,0,0), lineType=1)
    
    cv2.imwrite(save_img_file, show_img)
    return None

