## 3DUNet implemented with pytorch

## Introduction
The repository is a 3DUNet implemented with pytorch, referring to this [project](https://github.com/panxiaobai/lits_pytorch). I have redesigned the code structure and used the model to perform liver and tumor segmentation on the lits2017 dataset.  
paper: [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf)
#### requirement:  
```angular2
pytorch >= 1.1.0
torchvision
SimpleITK
Tensorboard
Scipy
```
## Quickly Start
### 1) LITS2017 dataset preprocessing: 
1. Download dataset from google drive: [Liver Tumor Segmentation Challenge.](https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE)  
Or from my share: https://pan.baidu.com/s/1WgP2Ttxn_CV-yRT4UyqHWw 
Extraction code：hfl8   
2. Then you need decompress the data set and put the volume data and segmentation labels into different local folders, such as `./dataset/data` and `./dataset/label`
3. Finally, you need to change the root path of the volume data and segmentation labels in `preprocess/preprocess_LiTS.py`, such as:
```
    row_dataset_path = './dataset/'  # path of origin dataset
    fixed_dataset_path = './fixed/'  # path of fixed(preprocessed) dataset
```   
4. Run `python preprocess/preprocess_LiTS.py`   
If nothing goes wrong, you can see the following files in the dir `./fixed`
```angular2
│  test_name_list.txt
│  train_name_list.txt
│  val_name_list.txt
│
├─data
│      volume-0.nii
│      volume-1.nii
│      volume-10.nii
│      ...
└─label
        segmentation-0.nii
        segmentation-1.nii
        segmentation-10.nii
        ...
```  
### 2) Training 3DUNet
1. Firstly, you should change the some parameters in `config.py`,especially, please set `--dataset_path` to `./fixed`  
All parameters are commented in the file `config.py`  
2. Secondely,run `python train.py`  
---
In addition, during the training process you will find that loading train data is time-consuming, you can use `train_faster.py` to train model. `train_faster.py` calls `./dataset/dataset2_lits.py`, which will crop multiple training samples from an input sample to form a batch for quick training.    
If you have any suggestions or questions, you can open an issue to communicate with me.  