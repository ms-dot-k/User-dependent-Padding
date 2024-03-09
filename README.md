# Speaker-adaptive Lip Reading with User-dependent Padding

This repository contains the PyTorch implementation of the following paper:
> **Speaker-adaptive Lip Reading with User-dependent Padding**<br>
> Minsu Kim, Hyunjun Kim, and Yong Man Ro<br>
> \[[Paper](https://arxiv.org/abs/2208.04498)\]

<div align="center"><img width="75%" src="img/img.png?raw=true" /></div>

## Preparation

### Requirements
- python 3.7
- pytorch 1.6 ~ 1.8
- torchvision
- ffmpeg
- av
- tensorboard
- pillow

### Datasets
LRW dataset can be downloaded from the below link.
- https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

The speaker annotations can be found in './data/LRW_ID_#.txt' or in this \[[repository](https://github.com/ms-dot-k/LRW_ID)\]

The pre-processing will be done in the data loader.<br>
The video is cropped with the bounding box \[x1:59, y1:95, x2:195, y2:231\].

### Preparing Baseline Model
You can download the Pretrained Baseline model. <br>
Put the ckpt in './data/checkpoints/Base/'

**Pretrained Baseline model**
- https://drive.google.com/file/d/1-3stMp4B_nTN2cmZW90z6o0cjTIv5Nsc/view?usp=sharing

|       Architecture      |   Acc.   |
|:-----------------------:|:--------:|
|Resnet18 + MS-TCN   |   85.847   |

or you can train your own baseline by using 'train.py'

## Speaker Adaptation of the Baseline Model by using User-Dependent Padding (UDP)
To speaker adapt the model, run following command:
```shell
# One GPU Training example for LRW
python train_udp.py \
--lrw 'enter_data_path' \
--checkpoint './data/checkpoints/Base/Baseline_85.847.ckpt' \
--batch_size 55 \
--checkpoint_dir 'checkpoint_path' \
--total_step 300
--subject 0 (~19)
--adapt_min 1 (1,3,5)
--fold 1
--gpu 0
```

Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint_dir`: the location for saving checkpoint
- `--checkpoint`: the checkpoint file
- `--batch_size`: batch size
- `--subject`: speaker id to be used for adaptation
- `--adapt_min`: length of adaptation data
- `--distributed`: Use DataDistributedParallel  
- `--dataparallel`: Use DataParallel
- `--gpu`: gpu for using `--lr`: learning rate 
- Refer to `train_udp.py` for the other training parameters

## Testing the Model
To test the model with UDP, run following command:
```shell
# Testing example for LRW
python test_udp.py \
--lrw 'enter_data_path' \
--checkpoint './data/checkpoints/Base/Baseline_85.847.ckpt' \
--checkpoint_udp 'checkpoint of trained user dependent padding' \
--batch_size 80 \
--subject 0 (~19) \
--gpu 0
```

Descriptions of training parameters are as follows:
- `--lrw`: training dataset location (lrw)
- `--checkpoint`: the checkpoint of baseline model
- `--checkpoint_udp`: the checkpoint of user dependent padding
- `--subject`: speaker id to be used for testing
- `--batch_size`: batch size
- `--gpu`: gpu for using
- Refer to `test_udp.py` for the other testing parameters

## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{kim2022speaker,
  title={Speaker-adaptive Lip Reading with User-dependent Padding},
  author={Kim, Minsu and Kim, Hyunjun and Ro, Yong Man},
  booktitle={European Conference on Computer Vision},
  pages={576--593},
  year={2022},
  organization={Springer}
}
```
