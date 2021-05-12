# Fully Convolutional One-Stage Object Detection (FCOS)
This repository contains my implementation of the Fully Convolutional One-Stage Object Detection([FCOS](https://arxiv.org/abs/1904.01355)) object detection architecture. Please note that the codes in this repository is still work-in-progress.

## Architecture of FCOS
As shown in Fig. 1, the FCOS architecture includes the Feature Pyramid Network (FPN) but adds separate classification and regression heads. In this code, the original FCOS algorithm is modified to follow a regression similar to that of [YOLOv3](https://arxiv.org/abs/1804.02767). This code also one target label in its corresponding feature map (P3 to P7) based on the centroid of the bounding box. The input image is also resized to dimensions of 512 by 512 or 640 by 640, depending on the processing capabilities of the GPU hardware. In general, an Nvidia Quadro P1000 graphics card is able to train on a 512 by 512 image using the COCO dataset.

![FCOS Architecture](FCOS_architecture.JPG)
Fig. 1: The original FCOS Architecture (as shown in the [FCOS](https://arxiv.org/abs/1904.01355) paper).

## Dataset
The model formats the bounding boxes in the VOC format with normalised coordinates (xmin, ymin, xmax, ymax) and generates the ground truth labels accordingly. For the COCO dataset, run
```
python process_COCO_annotations_fcos.py
python format_COCO_fcos.py
```
to obtain the normalised bounding boxes in the relevant format, followed by
```
python train_fcos_coco.py
```
to train the model.

## Training
The model is trained using SGD with a momentum of 0.9 and a batch size of 16. The training schedule follows that in the paper, and horizontal flipping is applied as the only data augmentation step.

## Inference
To perform inference, run
```
python infer_fcos_coco.py -t 0.25 -u 0.25 -s false -i kite.jpg
```
which generates the heatmap of detected objects as well as the boxes of the detected objects.

