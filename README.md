## Region-Aware Exposure Consistency Network for Mixed Exposure Correction (AAAI 2024)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2402.18217)

<hr />

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
If you find this work useful for your research, please consider citing:
``` 
@inproceedings{DBLP:conf/aaai/LiuFWM24,
  author       = {Jin Liu and
                  Huiyuan Fu and
                  Chuanming Wang and
                  Huadong Ma},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {Region-Aware Exposure Consistency Network for Mixed Exposure Correction},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {3648--3656},
  publisher    = {{AAAI} Press},
  year         = {2024},
  url          = {https://doi.org/10.1609/aaai.v38i4.28154},
  doi          = {10.1609/AAAI.V38I4.28154}
}
```

#### Acknowledgements
This repository is based on [ENC](https://github.com/KevinJ-Huang/ExposureNorm-Compensation) - special thanks to their code!
