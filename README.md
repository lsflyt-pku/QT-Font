# QT-Font
This repo is the official implementation of QT-Font: High-efficiency Font Synthesis via Quadtree-based Diffusion Models (SIGGRAPH 2024)

## Prerequisites
* Python
* PyTorch
* ocnn (which is modified to 2d in ocnn_2d)

## Data
We provide an example of the dataset in [PKU Disk](https://disk.pku.edu.cn/link/AA0B4605C580324595987B95091EDA5CBD). You can unzip it to the "data" directory.

Data organization referenced [VQ-Font](https://github.com/awei669/VQ-Font).

## Train
Run python main.py --config=#path_of_config

example: python main.py --config=config/chinesefont_train.yaml

## Test
Run python main.py --config=#path_of_config

example: python main.py --config=config/chinesefont_test.yaml

## Acknowledgement

This project is based on [ocnn-pytorch](https://ocnn-pytorch.readthedocs.io/en/latest/index.html) and [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN).
