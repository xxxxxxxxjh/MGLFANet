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

- **LEVIR-CD **: [DropBox](https://www.dropbox.com/scl/fi/uvp5q311jul5hzrvoivte/LEVIR-CD-256.zip?rlkey=3uahso53jvdmjfvw7fbotwb36&e=1&dl=0)
- **WHU-CD **: [DropBox](https://www.dropbox.com/scl/fi/8gczkg78fh95yofq5bs7p/WHU.zip?rlkey=05bpczx0gdp99hl6o2xr1zvyj&e=1&dl=0)
- **DSIFN-CD **: [DropBox](https://www.dropbox.com/scl/fi/ydj4u2shjp02n5q249au3/DSIFN.zip?rlkey=e0fa7iekeijos7o5xxbyy6bi4&e=1&dl=0)


# Train/Test

python main_cd.py/python eval_cd.py

