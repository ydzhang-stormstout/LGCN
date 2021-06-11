Lorentzian Graph Convolutional Networks
==================================================

The code containing an implementation of Lorentzian Graph Convolutional Networks (LGCN). The code is built upon HGCN (http://web.stanford.edu/~chami/files/hgcn.pdf)

## Requirements

- Python 3.7
- Pytorch >= 1.1
- numpy
- scikit-learn
- networkx
- tqdm

A recipe about installing the requirements is provided in `install.sh`.

## Usage

### ```train.py```

This script trains models for link prediction tasks. Please refer TensorFlow version for node classification task.

```
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --dropout DROPOUT     dropout probability
  --cuda CUDA           which cuda device to use (-1 for cpu training)
  --epochs EPOCHS       maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        l2 regularization strength
  --optimizer OPTIMIZER
                        which optimizer to use, can be any of [Adam,
                        RiemannianAdam]
  --momentum MOMENTUM   momentum in optimizer
  --patience PATIENCE   patience for early stopping
  --seed SEED           seed for training
  --log-freq LOG_FREQ   how often to compute print train/val metrics (in
                        epochs)
  --eval-freq EVAL_FREQ
                        how often to compute val metrics (in epochs)
  --save SAVE           1 to save model and logs and 0 otherwise
  --save-dir SAVE_DIR   path to save training logs and model weights 
  --sweep-c SWEEP_C
  --lr-reduce-freq LR_REDUCE_FREQ
                        reduce lr every lr-reduce-freq or None to keep lr
                        constant
  --gamma GAMMA         gamma for lr scheduler
  --print-epoch PRINT_EPOCH
  --grad-clip GRAD_CLIP
                        max norm for gradient clipping, or None for no
                        gradient clipping
  --min-epochs MIN_EPOCHS
                        do not early stop before min-epochs
  --task TASK           which tasks to train on, can be any of [lp, nc]
  --model MODEL         which encoder to use, can be any of [Shallow, LGCN]
  --dim DIM             embedding dimension
  --manifold MANIFOLD   which manifold to use, can be any of [Euclidean,
                        Lorentzian]
  --c C                 hyperbolic radius, set to None for trainable curvature
  --r R                 fermi-dirac decoder parameter for lp
  --t T                 fermi-dirac decoder parameter for lp
  --pretrained-embeddings PRETRAINED_EMBEDDINGS
                        path to pretrained embeddings (.npy file) for Shallow
                        node classification
  --pos-weight POS_WEIGHT
                        whether to upweight positive class in node
                        classification tasks
  --num-layers NUM_LAYERS
                        number of hidden layers in encoder
  --bias BIAS           whether to use bias (1) or not (0)
  --act ACT             which activation function to use (or None for no
                        activation)
  --n-heads N_HEADS     number of attention heads for graph attention
                        networks, must be a divisor dim
  --alpha ALPHA         alpha for leakyrelu in graph attention networks
  --double-precision DOUBLE_PRECISION
                        whether to use double precision
  --dataset DATASET     which dataset to use
  --val-prop VAL_PROP   proportion of validation edges for link prediction
  --test-prop TEST_PROP
                        proportion of test edges for link prediction
  --use-feats USE_FEATS
                        whether to use node features or not
  --normalize-feats NORMALIZE_FEATS
                        whether to normalize input node features
  --normalize-adj NORMALIZE_ADJ
                        whether to row-normalize the adjacency matrix
  --split-seed SPLIT_SEED
                        seed for data splits (train/test/val)
```

## Examples

#### Link prediction

```python train.py --task lp --dataset pubmed --model LGCN --lr 0.005 --dim 64 --num-layers 2 --act relu --dropout 0.5 --weight-decay 0.0 --manifold Lorentzian --log-freq 5 --cuda 0 --c None``` 

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
