# FPConv
FPConv: Learning Local Flattening for Point Convolution

### Installation

This code has been tested with Python 3.6, PyTorch 1.2.0, CUDA 10.0 and CUDNN 7.4 on Ubuntu 18.04. 

Firstly, install [pointnet2](https://github.com/sshaoshuai/Pointnet2.PyTorch) by running the following command:

```shell
cd fpconv/pointnet2
python setup.py install
```

You may also need to install plyfile and pickle for preprocessing.

### Usage

You need to edit the global configuration file `config.json` before training.

```json
{
    "version": "0.1",
    "scannet_raw": "<path_to_scannet_raw>",
    "scannet_pickle": "<path_to_save_scannet_pickle>",
    "scene_list": "<path_to_this_repo>/utils/scannet_datalist",
}
```

#### ScanNet

Generate training data by runing:

```shell
cd utils
python gen_pickle.py
```

__1. Train__

Run the following script to start the training:

```
./train_scannet.sh
```

We train our model with 2 Titan Xp GPUs with batch size of 12. If you don't have enough GPUs for training, please reduce `batch_size` to 6 for single GPU.

__2. Evaluate__

Run the following script to evaluate model on evaluation dataset (you may need to modify the `epoch` in `./test_scannet.sh`):

```shell
./test_scannet.sh
```

__Note__:

Final evaluation (by running `./test_scannet.sh`) is conducted on full point cloud, while evaluation during the training phase is conducted on randomly sampled points in each block of input scene.



