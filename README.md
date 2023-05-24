# Pure Pytorch Meta-Dataset

This repository contains Pytorch implementation of Meta-Dataset without any component of TensorFlow, as well as implementation of our ICML 2023 paper: A Closer Look at Few-shot Classification Again. Some features of our implementation:

1.  Unlike original Meta-Dataset, No dataset conversion to TFrecord is needed; instead we use raw images. This is beneficial for anyone who wants to inspect the dataset manually. 
2.  Unlike other versions of pytorch implementations of Meta-Dataset, we support multi-dataset training for both episodic/non-episodic methods.
3.  We completely fix the data shuffling problem arised in the original tensorflow implementation of Meta-Dataset (see [issue #54](https://github.com/google-research/meta-dataset/issues/54)), which strongly influences the evaluation results of ILSVRC, Aircraft, Traffic Signs, MSCOCO and Fungi. We solve this problem by randomly selecting data from all images in every episode, which cannot be done easily using the original Meta-Dataset implementation.


# Installation
Install packages using pip:
```bash
$ pip install -r requirements.txt
```

If you want to use pre-trained CLIP as the feature extractor, then run

```bash
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

# Pre-trained Models
All pre-trained models of Table 1 in the ICML 2023 paper can be found at [here](https://drive.google.com/drive/folders/18XhJxBFP_A2SDf7aSKtU-mpyvVhOTczW).

# Dataset Preparation
First, download all datasets of Meta-Dataset.
- ILSVRC 2012: 
1. Download `ilsvrc2012_img_train.tar`, from the [ILSVRC2012 website](http://www.image-net.org/challenges/LSVRC/2012/index)
2. Extract it into `ILSVRC2012_img_train/`, which should contain 1000 files, named `n????????.tar`
3. Extract each of `ILSVRC2012_img_train/n????????.tar` in its own directory
    (expected time: \~30 minutes), for instance:

    ```bash
    for FILE in *.tar;
    do
      mkdir ${FILE/.tar/};
      cd ${FILE/.tar/};
      tar xvf ../$FILE;
      cd ..;
    done
    ```

- Omniglot: 

1. Download
    [`images_background.zip`](https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip)
    and
    [`images_evaluation.zip`](https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip), and extract them into the same directory.

2. Launch the conversion script:
    ```bash
    python -m prepare_datasets.py \
        --data_src_path=<omniglot_source_path> \
        --data_dst_path=<omniglot_target_path> \
        --process_omniglot=1
    ```
    where `omniglot_source_path` refers to the directory of raw dataset, and `omniglot_target_path` refers to the directory of new converted dataset.

- Aircraft: 

1. Download [`fgvc-aircraft-2013b.tar.gz`](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz) and extract it into a directory.
2. Launch the conversion script:
    ```bash
    python -m prepare_datasets.py \
        --data_src_path=<aircraft_source_path> \
        --data_dst_path=<aircraft_target_path> \
        --process_aircraft=1
    ```
  where `aircraft_source_path` refers to the directory of raw dataset, and `aircraft_target_path` refers to the directory of new converted dataset.

- CUB: Download
    [`CUB_200_2011.tgz`](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and extract it.

- DTD: Download
    [`dtd-r1.0.1.tar.gz`](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz) and extract it.

- Quick Draw: 

1. Download all 345 `.npy` files hosted on
    [Google Cloud](https://console.cloud.google.com/storage/quickdraw_dataset/full/numpy_bitmap). You can use
        [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install#install)
        to download them to `quickdraw/`:

        ```bash
        gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy $DATASRC/quickdraw
        ```
2.  Launch the conversion script:
    ```bash
    python -m prepare_datasets.py \
        --data_src_path=<quickdraw_source_path> \
        --data_dst_path=<quickdraw_target_path> \
        --process_DuickD=1
    ```
    where `quickdraw_source_path` refers to the directory of raw dataset, and `quickdraw_target_path` refers to the directory of new converted dataset.

- Fungi: Download
    [`fungi_train_val.tgz`](https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz)
    and
    [`train_val_annotations.tgz`](https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz), then extract them into the same directory. It should contain one
    `images/` directory, as well as `train.json` and `val.json`.

- VGG Flower: Download
    [`102flowers.tgz`](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
    and
    [`imagelabels.mat`](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat), then extract `102flowers.tgz`, it will create a `jpg/` sub-directory

- Traffic Signs: 

1. Download
    [`GTSRB_Final_Training_Images.zip`](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip)
    If the link happens to be broken, browse the GTSRB dataset [website](http://benchmark.ini.rub.de) for more information. Then extract it, which will create a `GTSRB/` sub-directory.

2.  Launch the conversion script:
    ```bash
    python -m prepare_datasets.py \
        --data_src_path=<traffic_source_path> \
        --data_dst_path=<traffic_target_path> \
        --process_traffic=1
    ```
    where `traffic_source_path` refers to the directory of raw dataset, and `traffic_target_path` refers to the directory of new converted dataset.

- MSCOCO:

1. Download [`train2017.zip`](http://images.cocodataset.org/zips/train2017.zip) and
        [`annotations_trainval2017.zip`](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
        and extract them into the same directory.
2.  Launch the conversion script:
    ```bash
    python -m prepare_datasets.py \
        --data_src_path=<coco_source_path> \
        --data_dst_path=<coco_target_path> \
        --process_coco=1
    ```
    where `coco_source_path` refers to the directory of raw dataset, and `coco_target_path` refers to the directory of new converted dataset.

- MNIST:

1. Download [`t10k-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) and [`t10k-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz) into the same directory. 
2. Launch the conversion script:
    ```bash
    python -m prepare_datasets.py \
        --data_src_path=<mnist_source_path> \
        --data_dst_path=<mnist_target_path> \
        --process_mnist=1
    ```
    where `mnist_source_path` refers to the directory of raw dataset, and `mnist_target_path` refers to the directory of new converted dataset.

- CIFAR10: 

1. Download [`cifar-10-python.tar.gz`](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract it.
2. Launch the conversion script:
    ```bash
    python -m prepare_datasets.py \
        --data_src_path=<cifar10_source_path> \
        --data_dst_path=<cifar10_target_path> \
        --process_CIFAR10=1
    ```
    where `cifar10_source_path` refers to the directory of raw dataset, and `cifar10_target_path` refers to the directory of new converted dataset.

- CIFAR100:

1. Download [`cifar-100-python.tar.gz`](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and extract it.
2. Launch the conversion script:
    ```bash
    python -m prepare_datasets.py \
        --data_src_path=<cifar100_source_path> \
        --data_dst_path=<cifar100_target_path> \
        --process_CIFAR100=1
    ```
    where `cifar100_source_path` refers to the directory of raw dataset, and `cifar100_target_path` refers to the directory of new converted dataset.

- miniImageNet (not included in Meta-Dataset): Download [miniImageNet.zip](https://drive.google.com/file/d/1QEbHFIOKIM9KmId175QaLK-r22kgd7br/view?usp=share_link).

# Training and Testing

Experiments are defined via [yaml](configs) files with the help of [YACS](https://github.com/rbgirshick/yacs) package, following [Swin Trasformer](https://github.com/microsoft/Swin-Transformer/blob/main). The basic configurations are defined in `config.py`, overwritten by yaml files. yaml files can be written by python files, and we give examples for training CE models, training PN models, testing a pre-trained model, and seaching for hyperparameters for finetuning methods in `write_yaml_CE.py`, `write_yaml_PN.py`, `write_yaml_test.py` and `write_yaml_search.py`, respectively. Exemplar running scripts can be found in `train.sh`.

# Citation

If you find our work useful in your research please consider citing:

```
@inproceedings{
Luo2023closerlook,
title={A Closer Look at Few-shot Classification Again},
author={Luo, Xu and Wu, Hao and Zhang, Ji and Gao, Lianli and Xu, Jing and Song, Jingkuan},
booktitle={International Conference on Machine Learning},
year={2023},
}
```

## Acknowlegements

Except for original Meta-Dataset repo, part of the code is from [SwinTransformer](https://github.com/microsoft/Swin-Transformer/blob/main), [DeepEMD](https://github.com/icoz69/DeepEMD), [RFS](https://github.com/WangYueFt/rfs), [LightningFSL](https://github.com/Frankluox/LightningFSL), [S2M2](https://github.com/nupurkmr9/S2M2_fewshot), [eTT](https://github.com/loadder/eTT_TMLR2022), [URL and TSA](https://github.com/VICO-UoE/URL/tree/master) and [BiT](https://github.com/google-research/big_transfer).


