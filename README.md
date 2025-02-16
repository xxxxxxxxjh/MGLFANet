# MGLFA-Net: Multi-Scale Global-Local Feature Aggregation Network for Remote Sensing Change Detection

# Change Detection Dataset

## Requirements


Please see [`requirements.txt`](requirements.txt) for all the other requirements.



# 💬 Dataset Preparation

# 📂 Data structure

"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""

- **A** : images of t1 phase;
- **B** : images of t2 phase;
- **label** : label maps;
- **list** : contains `train.txt`, `val.txt` and `test.txt`,  each file records the image names (XXX.png) in the change detection dataset.
