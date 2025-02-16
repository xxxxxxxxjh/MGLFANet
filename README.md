# MGLFA-Net: Multi-Scale Global-Local Feature Aggregation Network for Remote Sensing Change Detection



## Requirements


Please see [`requirements.txt`](requirements.txt) for all the other requirements.



# 💬 Dataset Preparation

  📂 Data structure

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

## Our Processed Dataset Download

### Download method #1

- **LEVIR-CD (2.3GB)**: [DropBox](https://www.dropbox.com/s/example_link1)
- **WHU-CD (1.82GB)**: [DropBox](https://www.dropbox.com/s/example_link2)
- **DSIFN-CD (3.38GB)**: [DropBox](https://www.dropbox.com/s/example_link3)


# Train/Test

python main_cd.py/python eval_cd.py

