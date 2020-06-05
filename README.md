# FPConv

Yiqun Lin, Zizheng Yan, Haibin Huang, Dong Du, Ligang Liu, Shuguang Cui, Xiaoguang Han, "FPConv: Learning Local Flattening for Point Convolution", CVPR 2020 [[paper]](https://arxiv.org/abs/2002.10701)

```
@InProceedings{lin_fpconv_cvpr2020,
    author = {Yiqun Lin, Zizheng Yan, Haibin Huang, Dong Du, Ligang Liu, Shuguang Cui, Xiaoguang Han},
    title = {FPConv: Learning Local Flattening for Point Convolution},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

## Introduction

We introduce FPConv, a novel surface-style convolution operator designed for 3D point cloud analysis. Unlike previous methods, FPConv doesn't require transforming to intermediate representation like 3D grid or graph and directly works on surface geometry of point cloud. To be more specific, for each point, FPConv performs a local flattening by automatically learning a weight map to softly project surrounding points onto a 2D grid. Regular 2D convolution can thus be applied for efficient feature learning. FPConv can be easily integrated into various network architectures for tasks like 3D object classification and 3D scene segmentation, and achieve comparable performance with existing volumetric-type convolutions. More importantly, our experiments also show that FPConv can be a complementary of volumetric convolutions and jointly training them can further boost overall performance into state-of-the-art results.

![fig3](figures/fig3.png)

## Installation

This code has been tested with Python 3.6, PyTorch 1.2.0, CUDA 10.0 and CUDNN 7.4 on Ubuntu 18.04. 

Firstly, install [pointnet2](https://github.com/sshaoshuai/Pointnet2.PyTorch) by running the following commands:

```shell
cd fpconv/pointnet2
python setup.py install
cd ../
```

You may also need to install plyfile and pickle for data preprocessing.

## Usage

Edit the global configuration file `config.json` before training.

```json
{
    "version": "0.0",
    "scannet_raw": "<path_to_this_repo>/dataset/scannet_v2",
    "scannet_pickle": "<path_to_this_repo>/dataset/scannet_pickles",
    "scene_list": "<path_to_this_repo>/utils/scannet_datalist"
}
```

### Semantic Segmentation on ScanNet

__0. Baseline__

Tested on eval split of ScanNet. (see `./utils/scannet_datalist/scannetv2_eval.txt`)

| Model         | mIoU | mA   | oA   | download                                                     |
| ------------- | ---- | ---- | ---- | ------------------------------------------------------------ |
| fpcnn_scannet | 64.4 | 76.4 | 85.8 | [ckpt-17.8M](https://drive.google.com/file/d/1jR-m3bx2tGo9oV4ULdaYr-woSiel659T/view?usp=sharing) |

__1. Preprocessing__

Download ScanNet v2 dataset to `./dataset/scannet_v2`. Only `_vh_clean_2.ply` and `_vh_clean_2.labels.ply` should be downloaded for each scene. Then run following commands for data pre-processing.

```shell
cd utils
python gen_pickle.py
```

It will generate 3 pickle files (`scannet_<split>_rgb21c_pointid.pickle`) for 3 splits (train, eval, test) respectively. We will provide pre-processed ScanNet v2 dataset later. The `./dataset` folder should be organized as follows:

```
FPConv
├── dataset
│   ├── scannet_v2
│   │  ├── scans
│   │  ├── scans_test
│   ├── scannet_pickles
│   │  ├── scannet_train_rgb21c_pointid.pickle
│   │  ├── scannet_eval_rgb21c_pointid.pickle
│   │  ├── scannet_test_rgb21c_pointid.pickle
```

__2. Training__

Run the following command to start the training. Output (logs) will be redirected to `./logs/fp_test/nohup.log`.

```shell
bash train_scannet.sh
```

We trained our model with 2 Titan Xp GPUs with batch size of 12. If you don't have enough GPUs for training, please reduce `batch_size` to 6 for single GPU.

__3. Evaluation__

Run the following command to evaluate model on evaluation dataset (you may need to modify the `epoch` in `./test_scannet.sh`). Output (logs) will be redirected to `./test/fp_test_240.log`.

```shell
bash test_scannet.sh
```

__Note__: Final evaluation (by running `./test_scannet.sh`) is conducted on full point cloud, while evaluation during the training phase is conducted on randomly sampled points in each block of input scene.

### Semantic Segmentation on S3DIS

Codes will be released soon.

## Acknowledgement

- [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch): PyTorch implementation of PointNet++.

## License

This repository is released under MIT License (see LICENSE file for details).

