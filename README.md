## Region-Aware Exposure Consistency Network for Mixed Exposure Correction (AAAI 2024)

Jin Liu, Huiyuan Fu, Chuanming Wang, Huadong Ma

Beijing University of Posts and Telecommunications (BUPT)

The paper can be found here: https://arxiv.org/abs/2402.18217

#### 1. Introduction
This repository is the official implementation of the RECNet, where more implementation details are presented.

#### 2. Requirements
```
python=3.8.16
pytorch=2.0.1
torchvision=0.8
cuda=11.7
opencv-python
```

#### 3. Dataset Preparation
Refer to [ENC](https://github.com/KevinJ-Huang/ExposureNorm-Compensation) for details.

#### 4. Testing
```
bash scripts/test.sh
```
#### 5. Training
```a
bash scripts/train.sh
```

#### Citation
If you find this work useful for your research, please cite our paper:
``` 
@article{liu2024region,
  title={Region-Aware Exposure Consistency Network for Mixed Exposure Correction},
  author={Liu, Jin and Fu, Huiyuan and Wang, Chuanming and Ma, Huadong},
  journal={arXiv preprint arXiv:2402.18217},
  year={2024}
}
```

#### Acknowledgements
This repository is based on [ENC](https://github.com/KevinJ-Huang/ExposureNorm-Compensation) - special thanks to their code!
