# FacialMaskDetetor with Modified YOLOv4
Bachelor Thesis at NUAA, guided by Prof. Chaoying Tang.
---

## Introduction

This is the code for my Bachelor Thesis: YOLOv4 Based Facial Mask Detector. Most of facial mask detectors are only able to classify 2 classes: either "masked" or "face", and tend misclassify occlusion objects on human face with the "masked" label. It is contradictory to the sanity requirement of containing the infectious virus. Thus, we need a more sensitive detector for the "occlusion" class, which is defined with objects in front of the face but not facial masks.

this work deals with different occlusions

- Input: RGB image / camera

- Output: bounding boxes with classification (masked, face, occlusion)

  



## Environments

`torch==1.2`

To install requirements, create a virtual environment with conda, then install dependencies in `requirements.txt`. 

```
cd <YOUR DIR OF REPO>/MaskDetector
conda create -n maskdetector python=3.6
conda install --yes --file requirements.txt
```

Please pay attention to your `CUDA` version before installing pytorch!



## Works

- [x] Backbone: DarkNet53
- [x] Feature Pyramids: SPP，PAN
- [x] Training tricks：Mosaic data augmentation、Label Smoothing、CIOU、Cosine Learning Rate scheduler
- [x] Activation function: Mish



## Set Training Tricks

In `train.py` you could choose some training tricks implemented in the code:

1. `mosaic`:  a data augmentation method
2. `cosine_scheduler`: a learning rate scheduler to use cosine learning rates
3. `label_smoothing`: use label smoothing technique to smooth the output label 



## Pretrained Weights

Due to storage limits and privacy concerns, the weights are not provided. You could load a pretrained yolov4 weights. 



## Training Processes
1. Training samples are saved in the form of VOC datasets

2. Save Ground Truth labels to `VOCdevkit/VOC2007/Annotations`

3. Save images to  `VOCdevkit/VOC2007/JPEGImages`

4. run `voc2yolo3.py` to generate corresponding txt files for training, validation, and testing

5. run `voc_annotation.py` after changing class names, which will generate the `2007_train.txt` file

6. run `train.py`

   

### Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
