# Self-Driving Car Computer Vision Projects

</br>
<div align="center">
<a href="https://www.python.org/"><img src="./readme-content/Python.png" width="75" height="75"></a>
<a href="https://jupyter.org/"><img src="./readme-content/Jupyter.png" width="70" height="75"></a>
<a href="https://numpy.org/"><img src="./readme-content/Numpy.png" width="75" height="75"></a>
<a href="https://www.tensorflow.org/"><img src="./readme-content/Tensorflow.png" width="75" height="75"></a>
<a href="https://opencv.org/"><img src="./readme-content/OpenCV.png" width="75" height="75"></a>
</div>
</br>

## Overview

- This repository is the result of following [this tutorial](https://www.youtube.com/watch?v=cPOtULagNnI) from Freecodecamp regarding how machine learning is applied in self-driving cars
- The purpose of following this tutorial was to strengthen my computer vision skills and to learn how they can be applied to an area that I am interested in, which is self-driving cars
- The content of this repository is split into seven sections; Road Segmentation with a Fully Convolutional Network, 2D Object Detection using YOLO, Object Tracking using Deep Sort, Homogenous Transforms using KITTI 3D Data Visualisation, Multi-Task Learning using a Multi Task Attention Network (MTAN), 3D Object Detection using SFA 3D, and Camera to Bird's Eye View using UNetXST
- The datasets used for these projects were not included in this repository due to their file size, but they can be found in the tutorial linked above via Kaggle

## Road Segmentation | Fully Convolutional Network

This project involves using [semantic segmentation](https://paperswithcode.com/task/semantic-segmentation) to identify the road that a self-driving car would be travelling on and distinguish it from the rest of its video input. In this project, this is facilitated by using a [Fully Convolutional Network](https://paperswithcode.com/method/fcn), which are a type of neural network architecture designed for semantic segmentation tasks, where the goal is to classify and assign a label to each pixel in an input image. Unlike traditional convolutional neural networks (CNNs) that output a fixed-size prediction, FCNs preserve spatial information by using transposed convolutions to upsample the feature maps, allowing them to generate dense pixel-wise predictions. The model used for this project was an adapted version of the [VGG16 model](https://datagen.tech/guides/computer-vision/vgg16/). Initially, the model was trained on image data, and then the trained model was tested on video input.

## 2D Object Detection | YOLO

## Object Tracking | Deep SORT

## Homogenous Transformations | KITTI 3D Data Visualisation

## Multi Task Learning | Multi Task Attention Network (MTAN)

## 3D Object Detection | SFA 3D

## Camera to Bird's Eye View | UNetXST
