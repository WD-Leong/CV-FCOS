import numpy as np
import tensorflow as tf
from utils import swap_xy, convert_to_xywh

def _parse_image(filename):
    image_string  = tf.io.read_file(filename)
    image_decoded = \
        tf.image.decode_jpeg(image_string, channels=3)
    return tf.cast(image_decoded, tf.float32)

def resize_image(sample):
    image = _parse_image(sample["image"])
    image = tf.image.resize(
        image, [sample["min_side"], sample["min_side"]])
    image = image / 127.5 - 1.0
    
    bbox  = tf.cast(swap_xy(
        sample["objects"]["bbox"]), tf.float32)
    bbox = convert_to_xywh(bbox)
    class_id = tf.cast(
        sample["objects"]["label"], dtype=tf.int32)
    return image, bbox, class_id

def random_flip_horizontal(image, boxes, p_flip=0.5):
    """
    Flips image and boxes horizontally with 50% chance
    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.
    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) <= p_flip:
        tmp_box = boxes
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1.0-tmp_box[:, 2], tmp_box[:, 1], 
             1.0-tmp_box[:, 0], tmp_box[:, 3]], axis=-1)
    return image, boxes

def resize_and_pad_image(
    image, jitter=[640, 1024], min_side=800.0, 
    max_side=1333.0, stride=128.0, equal_dims=True):
    """
    Resizes and pads image while preserving aspect ratio.
    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`
    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.
    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(
        tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform(
            (), jitter[0], jitter[1], dtype=tf.float32)
    
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    
    new_shape = ratio * image_shape
    img_resized = tf.image.resize(
        image, tf.cast(new_shape, tf.int32))
    img_resized = img_resized / 127.5 - 1.0
    
    padded_dims = tf.cast(tf.math.ceil(
        new_shape/stride) * stride, dtype=tf.int32)
    if equal_dims:
        max_dims = np.max(padded_dims.numpy())
        padded_dims = [max_dims, max_dims]
    
    image_padded = tf.image.pad_to_bounding_box(
        img_resized, 0, 0, padded_dims[0], padded_dims[1])
    return image_padded, new_shape, ratio

def preprocess_data(sample, img_dims=384, pad_flag=True):
    """
    Applies preprocessing step to a single sample.
    Arguments:
      sample: A dict representing a single training sample.
    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: A tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    jitter = [sample["l_jitter"], sample["u_jitter"]]
    
    image = _parse_image(sample["image"])
    if not pad_flag:
        image = tf.image.resize(
            image, [img_dims, img_dims])
    
    bbox = tf.cast(
        sample["objects"]["bbox"], tf.float32)
    class_id = tf.cast(
        sample["objects"]["label"], dtype=tf.int32)
    
    image, bbox = random_flip_horizontal(image, bbox)
    if pad_flag:
        image, img_shp, ratio = \
            resize_and_pad_image(
                image, min_side=sample["min_side"], 
                max_side=sample["max_side"], jitter=jitter)
    else:
        image = image / 127.5 - 1.0
        img_shp = tf.cast([img_dims, img_dims], tf.float32)
    
    bbox = swap_xy(bbox)
    bbox = convert_to_xywh(bbox)
    bbox = bbox.numpy()
    return image, tf.constant(bbox), class_id, img_shp

