# MGLFA-Net: Multi-Scale Global-Local Feature Aggregation Network for Remote Sensing Change Detection

# Change Detection Dataset

## Requirements

Python 3.8.0 pytorch 1.10.1 torchvision 0.11.2 einops 0.3.2

r
复制
编辑

Please see [`requirements.txt`](requirements.txt) for all the other requirements.

## 💬 Dataset Preparation

### 📂 Data structure

Change detection data set with pixel-level binary labels; ├── A ├── B ├── label └── list

markdown
复制
编辑

- **A** : images of t1 phase;
- **B** : images of t2 phase;
- **label** : label maps;
- **list** : contains `train.txt`, `val.txt` and `test.txt`, each file records the image names (`XXX.p
