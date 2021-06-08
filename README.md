# Complex-YOLO-V3
Complete but Unofficial PyTorch Implementation of [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/pdf/1803.06199.pdf) with YoloV3


# Custom Dataset on Colab
## Data Preparation for Custom Dataset

#### Raw data
Follow the same structure data structure above
```
└── data/KITTI/object
       ├── training       
       |   ├── image_2
       |   |   ├── 001.png
       |   |   ├── 002.png
       |   |   └── 003.png
       |   ├── calib
       |   |   ├── 001.txt
       |   |   ├── 002.txt
       |   |   └── 003.txt
       |   ├── label_2
       |   |   ├── 001.txt
       |   |   ├── 002.txt
       |   |   └── 003.txt
       |   ├── velodyne
       |   |   ├── 001.bin
       |   |   ├── 002.bin
       |   |   └── 003.bin
       └── testing
           ├── image_2
           |   └── 001.png
           ├── calib
           |   └── 001.txt
           └── velodyne
               └── 001.bin       
```
#### Label
Upload your own `valid.txt`, `train.txt` and `test.txt` to have to following structure.
```
└── data/KITTI/ImageSets
       ├── train.txt   
       ├── valid.txt 
       ├── test.txt 
              
```
Your `.txt` file should contains the name of the images file
For instance, continue with the example in the raw dataset.
Your `train.txt` could be:
```
001
002
```

Your `valid.txt` could be:
```
003
```

Your `test.txt` could be:
```
001
```

#### Class label file
Make a file named `classes.names`, according to the classes of your dataset. It should be like this
```
First Class name
Second Class name
...
Car
Person
```

## Training on Colab
#### Make your zip datafile
Zip your `train.txt`, `test.txt`, `valid.txt` into `label.zip`.
Zip your `training` folder and `testing` folder into `data.zip`.

#### Training
If the notebook doesn't work, check in Colab to see if you have the folder
structure above
Follow the instruction on this notebook (Download this notebook and open it in Google Colab) [a relative link](Colab_Complex_YOLOv3_Training.ipynb)

## Inference/Testing on Colab
Follow this Colab Notebook here (Download this notebook and open it in Google Colab) [a relative link](Colab_Complex_YOLOv3_Testing.ipynb)


# Original README.md

## Installation
#### Clone the project and install requirements
    $ git clone https://github.com/ghimiredhikura/Complex-YOLOv3
    $ cd Complex-YOLO-V3/
    $ sudo pip install -r requirements.txt

## Quickstart

#### Download pretrained weights [[yolov3](https://drive.google.com/file/d/1e7PCqeV3tS68KtBIUbX34Uhy81XnsYZV/view), [tiny-yolov3](https://drive.google.com/file/d/19Qvpq2kQyjQ5uhQgi-wcWmSqFy4fcvny/view)]
    $ cd checkpoints/
    $ python download_weights.py
    
#### Test [without downloading dataset] 

    1. $ python test_detection.py --split=sample --folder=sampledata  
    2. $ python test_both_side_detection.py --split=sample --folder=sampledata

#### Demo Video [[Click to Play](https://www.youtube.com/watch?v=JzywsbuXFOg)]
[![complex-yolov3_gif][complex-yolov3_gif]](https://youtu.be/JzywsbuXFOg)

[//]: # (Image References)
[complex-yolov3_gif]: ./assets/complex-yolov3.gif

## Data Preparation

You can see `sampledata` folder in `data/KITTI/dataset` directory which can be used for testing this project without downloading KITTI dataset. However, if you want to train the model by yourself and check the mAP in validation set just follow the steps below.

#### Download the [3D KITTI detection dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 
1. Camera calibration matrices of object data set (16 MB)
2. Training labels of object data set (5 MB)
3. Velodyne point clouds (29 GB)
4. Left color images of object data set (12 GB)

Now you have to manage dataset directory structure. Place your dataset into `data` folder. Please make sure that you have the dataset directory structure as follows. 

```
└── data/KITTI/object
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   └── velodyne
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           └── velodyne           
```

The `train/valid` split of training dataset as well as `sample` and `test` dataset ids are in `data/KITTI/ImageSets` directory. From training set of 7481 images, 6000 images are used for training and remaining 1481 images are used for validation. The mAP results reported in this project are evaluated into this valid set with custom mAP evaluation script with 0.5 iou for each object class.

## Train
    $ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] 
                [--pretrained_weights PRETRAINED_WEIGHTS] 
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--multiscale_training MULTISCALE_TRAINING]

--Training log example--

    ---- [Epoch 0/300, Batch 250/1441] ----  
    +------------+--------------+--------------+--------------+  
    | Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |  
    +------------+--------------+--------------+--------------+  
    | grid_size  | 17           | 34           | 68           |  
    | loss       | 6.952686     | 5.046788     | 4.256296     |  
    | x          | 0.054503     | 0.047048     | 0.060234     |  
    | y          | 0.110871     | 0.059848     | 0.081368     |
    | w          | 0.101059     | 0.056696     | 0.022349     |
    | h          | 0.294365     | 0.230845     | 0.076873     |
    | im         | 0.215230     | 0.218564     | 0.184226     |
    | re         | 1.049812     | 0.883522     | 0.783887     |
    | conf       | 4.682138     | 3.265709     | 2.941420     |
    | cls        | 0.444707     | 0.284557     | 0.105938     |
    | cls_acc    | 67.74%       | 83.87%       | 96.77%       |
    | recall50   | 0.000000     | 0.129032     | 0.322581     |
    | recall75   | 0.000000     | 0.032258     | 0.032258     |
    | precision  | 0.000000     | 0.285714     | 0.133333     |
    | conf_obj   | 0.058708     | 0.248192     | 0.347815     |
    | conf_noobj | 0.014188     | 0.007680     | 0.010709     |
    +------------+--------------+--------------+--------------+
    Total loss 16.255769729614258
    ---- ETA 0:18:27.490254

## Test
    $ python test_detection.py
    $ python test_both_side_detection.py

## Evaluation
    $ python eval_mAP.py 

mAP (min. 50 IoU)

| Model/Class             | Car     | Pedestrian | Cyclist | Average |
| ----------------------- |:--------|:-----------|:--------|:--------|
| Complex-YOLO-v3         | 97.89   |82.71       |90.12    |90.24    |
| Complex-Tiny-YOLO-v3    | 95.91   |49.29       |78.75    |74.65    |


#### Results 
<p align="center"><img src="assets/result1.jpg" width="1246"\></p>
<p align="center"><img src="assets/result2.jpg" width="1246"\></p>
<p align="center"><img src="assets/result3.jpg" width="1246"\></p>

## Credit

1. Complex-YOLO: https://arxiv.org/pdf/1803.06199.pdf

YoloV3 Implementation is borrowed from:
1. https://github.com/eriklindernoren/PyTorch-YOLOv3

Point Cloud Preprocessing is based on:  
1. https://github.com/skyhehe123/VoxelNet-pytorch
2. https://github.com/dongwoohhh/MV3D-Pytorch
