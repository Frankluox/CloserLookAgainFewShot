# Adapted from SwinTransformer

import os
import yaml
from yacs.config import CfgNode as CN


# This file contains basic configuration (example of a 5-way 5-shot PN model trained only on ImageNet).
# all configurations can be overwritten by yaml files.

_C = CN()

# Base config files to inherit from
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True

# Number of data loading threads. 
_C.DATA.NUM_WORKERS = 8

# Input image size
_C.DATA.IMG_SIZE = 224


# ImageNet settings
_C.DATA.PATH_TO_WORDS = ""
_C.DATA.PATH_TO_IS_A = ""
_C.DATA.PATH_TO_NUM_LEAF_IMAGES = ""
_C.DATA.TRAIN_SPLIT_ONLY = False

_C.DATA.TRAIN = CN()

# Total batch size (number of tasks for episodic training/testing),
# could be overwritten by command line argument.
_C.DATA.TRAIN.BATCH_SIZE = 1

# all training dataset names
_C.DATA.TRAIN.DATASET_NAMES = ["ILSVRC"]
# correponding dataset roots
_C.DATA.TRAIN.DATASET_ROOTS = [""]

# correponding sampling frequencies of each dataset
_C.DATA.TRAIN.SAMPLING_FREQUENCY = [1.]

# whether to do episodic training
_C.DATA.TRAIN.IS_EPISODIC = True




# a 5-way 5-shot training example.


# the configuration for episodic training, ignored if non-episodic
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG = CN()

# number of sampled tasks per epoch
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_TASKS_PER_EPOCH = 1000

# Whether to use the original Meta-Dataset sampling, 1 for true and 0 for false
# (This is a fault of the original implementation of
# Meta-Dataset, and will influence results of ILSVRC, Aircraft, Traffic Signs, MSCOCO and Fungi. 
# To understand the details, e.g., see https://github.com/google-research/meta-dataset/issues/54 for traffic signs)
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.SEQUENTIAL_SAMPLING = 0

# The followings are for data sampling, strictly following the original Meta-Dataset settings.

#========================================================================================
# The following three hyperparameters are for sampling of fixed number of way/shot/query 
# fix ways if not set to None
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_WAYS = 5
# fix shots if not set to None
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_SUPPORT = 5
# fix query data per class if not set to None
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_QUERY = 15
#========================================================================================

#========================================================================================
# varied way/shot sampling 

# minimum ways to sample
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.MIN_WAYS = 5
# maximum ways to sample
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.MAX_WAYS_UPPER_BOUND = 50
# maximum query data per class to sample
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.MAX_NUM_QUERY = 15
# remove classes that do not have enough images
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.MIN_EXAMPLES_IN_CLASS = 0
# maximum total number of images in the support set
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.MAX_SUPPORT_SET_SIZE = 500
# maximum total number of images per class in the support set
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS = 100
# randomly decide contribution of each class to the support set
# see Appendix of the Meta-Dataset paper for detail
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.MIN_LOG_WEIGHT = -0.69314718055994529
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.MAX_LOG_WEIGHT = 0.69314718055994529
#========================================================================================


# other settings for ImageNet
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.USE_DAG_HIERARCHY = False

# other settings for Omniglot
_C.DATA.TRAIN.EPISODE_DESCR_CONFIG.USE_BILEVEL_HIERARCHY =  False


# whether to shuffle the data for non-episodic training
_C.DATA.TRAIN.SHUFFLE = True    


# Iterations per epoch for non-episodic training
# if None, an epoch ends when all data of the dataset has been sampled.
_C.DATA.TRAIN.ITERATION_PER_EPOCH = None



# The same as training
_C.DATA.VALID = CN()

_C.DATA.VALID.BATCH_SIZE = 2
_C.DATA.VALID.DATASET_NAMES = ["ILSVRC"]
_C.DATA.VALID.DATASET_ROOTS = [""]
_C.DATA.VALID.SAMPLING_FREQUENCY = [1.]
_C.DATA.VALID.IS_EPISODIC = True

# example of a varied-way varied-shot validation that contains 600 tasks
_C.DATA.VALID.EPISODE_DESCR_CONFIG = CN()
_C.DATA.VALID.EPISODE_DESCR_CONFIG.SEQUENTIAL_SAMPLING = 0
_C.DATA.VALID.EPISODE_DESCR_CONFIG.NUM_TASKS_PER_EPOCH = 600
_C.DATA.VALID.EPISODE_DESCR_CONFIG.NUM_WAYS = None
_C.DATA.VALID.EPISODE_DESCR_CONFIG.NUM_SUPPORT = None
_C.DATA.VALID.EPISODE_DESCR_CONFIG.NUM_QUERY = None
_C.DATA.VALID.EPISODE_DESCR_CONFIG.MIN_WAYS = 5
_C.DATA.VALID.EPISODE_DESCR_CONFIG.MAX_WAYS_UPPER_BOUND = 50
_C.DATA.VALID.EPISODE_DESCR_CONFIG.MAX_NUM_QUERY = 10
_C.DATA.VALID.EPISODE_DESCR_CONFIG.MIN_EXAMPLES_IN_CLASS = 0
_C.DATA.VALID.EPISODE_DESCR_CONFIG.MAX_SUPPORT_SET_SIZE = 500
_C.DATA.VALID.EPISODE_DESCR_CONFIG.MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS = 100
_C.DATA.VALID.EPISODE_DESCR_CONFIG.MIN_LOG_WEIGHT = -0.69314718055994529
_C.DATA.VALID.EPISODE_DESCR_CONFIG.MAX_LOG_WEIGHT = 0.69314718055994529
# only works for ImageNet
_C.DATA.VALID.EPISODE_DESCR_CONFIG.USE_DAG_HIERARCHY = True
# only works for Omniglot
_C.DATA.VALID.EPISODE_DESCR_CONFIG.USE_BILEVEL_HIERARCHY =  False


_C.DATA.VALID.SHUFFLE = False
_C.DATA.VALID.ITERATION_PER_EPOCH = None    

_C.DATA.TEST = CN()

_C.DATA.TEST.BATCH_SIZE = 8
_C.DATA.TEST.DATASET_NAMES = ["ILSVRC"]
_C.DATA.TEST.DATASET_ROOTS = [""]
_C.DATA.TEST.SAMPLING_FREQUENCY = [1.]
_C.DATA.TEST.IS_EPISODIC = True
_C.DATA.TEST.DATASET_IDENTIFICATION = False


_C.DATA.TEST.EPISODE_DESCR_CONFIG = CN()
_C.DATA.TEST.EPISODE_DESCR_CONFIG.SEQUENTIAL_SAMPLING = 0
_C.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_TASKS_PER_EPOCH = 2000
_C.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_WAYS = None
_C.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT = None
_C.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_QUERY = None
_C.DATA.TEST.EPISODE_DESCR_CONFIG.MIN_WAYS = 5
_C.DATA.TEST.EPISODE_DESCR_CONFIG.MAX_WAYS_UPPER_BOUND = 50
_C.DATA.TEST.EPISODE_DESCR_CONFIG.MAX_NUM_QUERY = 10
_C.DATA.TEST.EPISODE_DESCR_CONFIG.MIN_EXAMPLES_IN_CLASS = 0
_C.DATA.TEST.EPISODE_DESCR_CONFIG.MAX_SUPPORT_SET_SIZE = 500
_C.DATA.TEST.EPISODE_DESCR_CONFIG.MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS = 100
_C.DATA.TEST.EPISODE_DESCR_CONFIG.MIN_LOG_WEIGHT = -0.69314718055994529
_C.DATA.TEST.EPISODE_DESCR_CONFIG.MAX_LOG_WEIGHT = 0.69314718055994529
# only works for ImageNet
_C.DATA.TEST.EPISODE_DESCR_CONFIG.USE_DAG_HIERARCHY = True
# only works for Omniglot
_C.DATA.TEST.EPISODE_DESCR_CONFIG.USE_BILEVEL_HIERARCHY =  False

_C.DATA.TEST.SHUFFLE = False  
_C.DATA.TEST.ITERATION_PER_EPOCH = None 


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()

# whether to use color jitter 
# if not use, None; if use, fill in [brightness, contrast, saturation, hue, probability of applying]
_C.AUG.COLOR_JITTER = None

# whether to use gray scale
# if not use, None; if use, fill in probability of applying
_C.AUG.GRAY_SCALE  = None

# whether to use Gaussian blur
# if not use, None; if use, fill in probability of applying
_C.AUG.GAUSSIAN_BLUR = None

# the probability to apply horizontal flip
_C.AUG.FLIP = 0.5

# normalization, ImageNet by default
_C.AUG.MEAN = [0.485, 0.456, 0.406]
_C.AUG.STD = [0.229, 0.224, 0.225]

# whether to center crop images at test time
_C.AUG.TEST_CROP = False

# # -----------------------------------------------------------------------------
# # Model settings
# # -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type; should match the file name under 'models' directory
_C.MODEL.TYPE = 'Episodic_Model'
# Model name; used as the foler name of results
_C.MODEL.NAME = 'miniImageNet_Res50_PN'
# Backbone used; should match the file name under 'architectures/backbone' directory
_C.MODEL.BACKBONE = 'resnet50'

# all hyperparameters of constructing the backbone; arranged sequentially as a list
_C.MODEL.BACKBONE_HYPERPARAMETERS = []

# classifer used for episodic training/testing; 
# should match the file name under 'architectures/classifier' directory
_C.MODEL.CLASSIFIER = 'proto_head'

# all hyperparameters of constructing the classifer; arranged sequentially as a list
_C.MODEL.CLASSIFIER_PARAMETERS = []

# whether initialize from a pre-trained model
_C.MODEL.PRETRAINED = None

# resume training from a checkpoint if a non-empty path is given
_C.MODEL.RESUME = ''






# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------


_C.TRAIN = CN()

# total training epochs
_C.TRAIN.EPOCHS = 60

# which epoch to start with
_C.TRAIN.START_EPOCH = 0

# the number of epochs for warm-up
_C.TRAIN.WARMUP_EPOCHS = 0

# The base learning rate
_C.TRAIN.BASE_LR = 0.1

# The initial warm-up learning rate
_C.TRAIN.WARMUP_LR_INIT = 0.

# weight decay
_C.TRAIN.WEIGHT_DECAY = 5e-4

# whether to schedule the learning rate per step
_C.TRAIN.SCHEDULE_PER_STEP = True


# Optimizer
_C.TRAIN.OPTIMIZER = CN()

# optimizer type
_C.TRAIN.OPTIMIZER.NAME = "SGD"

# momentum for SGD
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Sheduler
_C.TRAIN.LR_SCHEDULER = CN()

# LR scheduler type
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'

# whether to resume from previous checkpoints automatically
_C.TRAIN.AUTO_RESUME = True


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# training or testing
_C.IS_TRAIN = 1
# Used GPU ID
_C.GPU_ID = 0
# seed
_C.SEED = 0
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to logging info
_C.PRINT_FREQ = 10
# The number of top K models to save
_C.SAVE_TOP_K_MODEL = 5

# specify specific epochs to save
_C.SAVE_EPOCHS = [20,40]


# -----------------------------------------------------------------------------
# for hyperparameter search
# -----------------------------------------------------------------------------

_C.SEARCH_HYPERPARAMETERS = CN()

_C.SEARCH_HYPERPARAMETERS.LR_BACKBONE_RANGE = None

_C.SEARCH_HYPERPARAMETERS.LR_HEAD_RANGE = None

_C.SEARCH_HYPERPARAMETERS.EPOCH_RANGE = None








def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    #递归找base file
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}') is not None:
            return True
        return False

    # merge from specific arguments
    if _check_args('is_train'):
        config.IS_TRAIN = args.is_train
    if _check_args('train_batch_size'):
        config.DATA.TRAIN.BATCH_SIZE = args.train_batch_size
    if _check_args('valid_batch_size'):
        config.DATA.VALID.BATCH_SIZE = args.valid_batch_size
    if _check_args('test_batch_size'):
        config.DATA.TEST.BATCH_SIZE = args.test_batch_size
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag


    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
