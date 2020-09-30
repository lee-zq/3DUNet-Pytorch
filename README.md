## 3DUNet implemented with pytorch

## Introduction
The repository is a 3DUNet implemented with pytorch, referring to 
this [project](https://github.com/panxiaobai/lits_pytorch).
 I have redesigned the code structure and used the model to perform liver and tumor segmentation on the lits2017 dataset.  
paper: [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf)
### Requirements:  
```angular2
pytorch >= 1.1.0
torchvision
SimpleITK
Tensorboard
Scipy
```
### Code Structure
```angular2
├── config.py        # Configuration information for training and testing
├── dataset          # Training and testing dataset
│   ├── dataset_lits_faster.py 
│   ├── dataset_lits.py
│   └── test_dataset.py
├── models           # Model design
│   ├── nn
│   └── Unet.py
├── output           # Trained model
├── preprocess
│   └── preprocess_LiTS.py
├── test.py          # Test code
├── train_faster.py  # Quick training code
├── train.py         # Standard training code
└── utils            # Some related tools
    ├── common.py
    ├── init_util.py
    ├── logger.py
    ├── metrics.py
```
## Quickly Start
### 1) LITS2017 dataset preprocessing: 
1. Download dataset from google drive: [Liver Tumor Segmentation Challenge.](https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE)  
Or from my share: https://pan.baidu.com/s/1WgP2Ttxn_CV-yRT4UyqHWw 
Extraction code：hfl8   
2. Then you need decompress the dataset. It is recommended to use batch1(0\~27) of the LiTS dataset as the testset
 and batch2(28\~130) as the trainset. Please put the volume data and segmentation labels of trainset and testset into different local folders, 
such as:  
```
raw_dataset:
    ├── LiTS_batch1  # (0~27)
    │   ├── data
    │   │   ├── volume-0.nii
    │   │   ├── volume-10.nii ...
    │   └── label
    │       ├── segmentation-0.nii
    │       ├── segmentation-10.nii ...
    │       
    ├── LiTS_batch2 # (28~130)
    │   ├── data
    │   │   ├── volume-28.nii
    │   │   ├── volume-29.nii ...
    │   └── label
    │       ├── segmentation-28.nii
    │       ├── segmentation-29.nii ...
```
3. Finally, you need to change the root path of the volume data and segmentation labels in `preprocess/preprocess_LiTS.py`, such as:
```
    row_dataset_path = './raw_dataset/LiTS_batch2/'  # path of origin dataset
    fixed_dataset_path = './fixed_data/'  # path of fixed(preprocessed) dataset
```   
4. Run `python preprocess/preprocess_LiTS.py`   
If nothing goes wrong, you can see the following files in the dir `./fixed_data`
```angular2
│  train_name_list.txt
│  val_name_list.txt
│
├─data
│      volume-28.nii
│      volume-29.nii
│      volume-30.nii
│      ...
└─label
        segmentation-28.nii
        segmentation-29.nii
        segmentation-30.nii
        ...
```  
### 2) Training 3DUNet
1. Firstly, you should change the some parameters in `config.py`,especially, please set `--dataset_path` to `./fixed_data`  
All parameters are commented in the file `config.py`. 
2. Secondely,run `python train.py --save model_name`  
3. Besides, you can observe the dice and loss during the training process 
in the browser through `tensorboard --logdir ./output/model_name`. 
---
In addition, during the training process you will 
find that loading train data is time-consuming, 
you can use `train_faster.py` to train model. `train_faster.py` calls `./dataset/dataset_lits_faster.py`,
 which will crop multiple training samples from an input sample to form a batch for quickly training.    
### 3) Testing  
run `test.py`  
Please pay attention to path of trained model and cut parameters in `test.py`.   
(Since the calculation of the 3D convolution operation is too large,
 I use a sliding window to block the input tensor before prediction, and then stitch the results to get the final result.
 The size of the sliding window can be set by yourself in `test.py`)  

After the test, you can get the test results in the corresponding folder:`./output/model_name/result`

You can also read my Chinese
 introduction about this [3DUNet project here](https://zhuanlan.zhihu.com/p/113318562).    
If you have any suggestions or questions, 
welcome to open an issue to communicate with me.  