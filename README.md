# iCLIP

This project is the official implementation of our paper 
[Interaction-Aware Prompting for Zero-Shot Spatio-Temporal Action Detection](https://arxiv.org/abs/2304.04688) (**ICCV Workshop 2023**), authored by Wei-Jhe Huang, Jheng-Hsien Yeh, Min-Hung Chen, Gueter Josmy Faure and Shang-Hong Lai. 

## Installation

You need first to install this project, please check [INSTALL.md](INSTALL.md)

## Data Preparation

To do training or inference on J-HMDB, please check [DATA.md](DATA.md)
for data preparation instructions. Instructions for other datasets coming soon.

## Model Zoo

Please see [MODEL_ZOO.md](MODEL_ZOO.md) for downloading models.

## Training and Inference

To do training or inference with iCLIP, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## Acknowledgement
We are very grateful to the authors of [AlphAction](https://github.com/MVIG-SJTU/AlphAction) and [HIT](https://github.com/joslefaure/HIT) for open-sourcing their code from which this repository is heavily sourced. If your find these researchs useful, please consider citing their paper as well.

```
@inproceedings{tang2020asynchronous,
  title={Asynchronous Interaction Aggregation for Action Detection},
  author={Tang, Jiajun and Xia, Jin and Mu, Xinzhi and Pang, Bo and Lu, Cewu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2020}
}
```
```
@InProceedings{Faure_2023_WACV,
    author    = {Faure, Gueter Josmy and Chen, Min-Hung and Lai, Shang-Hong},
    title     = {Holistic Interaction Transformer Network for Action Detection},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {3340-3350}
}
```