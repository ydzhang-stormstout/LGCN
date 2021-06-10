# Lorentzian Graph Convolutional Networks

The code containing a TensorFlow implementation of Lorentzian Graph Convolutional Networks (LGCN).
The code is built upon GAT (https://arxiv.org/abs/1710.10903)

## Overview
The repository is organised as follows:
- `data/` contains the dataset files;
- `models/` contains the implementation of the HAT (`sp_hgat.py`);
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`hlayers.py`);
    * preprocessing subroutines (`process.py`);

Finally, `hat.py` puts all of the above together and may be used to execute a full training run on Cora.

## Dependencies

The script has been tested running under Python 3.7, with the following packages installed:

- `numpy`
- `scipy`
- `networkx`
- `tensorflow-gpu==1.14.0`


## Reference
If you make advantage of the HAT model in your research, please cite the following in your manuscript:

```
@inproceedings{zhang2021lorentzian,
  title={Lorentzian Graph Convolutional Networks},
  author={Zhang, Yiding and Wang, Xiao and Shi, Chuan and Liu, Nian and Song, Guojie},
  booktitle={Proceedings of the Web Conference 2021},
  pages={1249--1261},
  year={2021}
```