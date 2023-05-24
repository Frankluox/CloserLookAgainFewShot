import yaml
import os

all_roots = {}
all_roots["ILSVRC"] = "/home/wuhao/data/imagenet/train" #0
all_roots["Omniglot"] = "../data/all_datasets/new_omniglot" #1
all_roots["Quick Draw"] = "../data/all_datasets/domainnet/quickdraw" #2
all_roots["Birds"] = "../data/all_datasets/CUB_200_2011" #3
all_roots["VGG Flower"] = "../data/all_datasets/vggflowers" #4
all_roots["Aircraft"] = "../data/all_datasets/aircraft"  #5
all_roots["Traffic Signs"] = "../data/traffic" #6
all_roots["MSCOCO"] = "../data/coco" #7
all_roots["Textures"] = "../data/all_datasets/dtd" #8
all_roots["Fungi"] = "../data/all_datasets/fungi" #9
all_roots["MNIST"] = "../data/all_datasets/mnist" #10
all_roots["CIFAR10"] = "../data/all_datasets/cifar10" #11
all_roots["CIFAR100"] = "../data/all_datasets/cifar100" #12
all_roots["miniImageNet"] = "/home/wuhao/data/mini_imagenet/images_imagefolder" #13

Data = {}

Data["DATA"] = {}

Data["DATA"]["PATH_TO_WORDS"] = "/home/luoxu/data/words.txt"
Data["DATA"]["PATH_TO_IS_A"] = "/home/luoxu/data/wordnet.is_a.txt"
Data["DATA"]["PATH_TO_NUM_LEAF_IMAGES"] = "data/ImageNet_num_images_perclass.json"

Data["IS_TRAIN"] = 0

names = list(all_roots.keys())
roots = list(all_roots.values())


Data["DATA"]["TEST"] = {}

Data["DATA"]["TEST"]["DATASET_ROOTS"] = [roots[6]]
Data["DATA"]["TEST"]["DATASET_NAMES"] = [names[6]]


# 5 way 1 shot example
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 1
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]
Data["DATA"]["TEST"]["BATCH_SIZE"] = 8

# 5 way 5 shot example
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"] = {}
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 5
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["USE_BILEVEL_HIERARCHY"] = False
# Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]
# Data["DATA"]["TEST"]["BATCH_SIZE"] = 8

# varied way varied shot example
# Data["DATA"]["TEST"]["BATCH_SIZE"] = 8



Data["OUTPUT"] = "../new_metadataset_result"
Data["MODEL"] = {}

Data["MODEL"]["NAME"] = "evaluation"
Data["GPU_ID"] = 2

# 1 if use sequential sampling in the oroginal biased Meta-Dataset sampling procedure, 0 unbiased.
# 1 can be used to re-implement the results in the ICML 2023 paper (except traffic signs); 0, however, is recommended for unbiased results
Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 1

Data["AUG"] = {}

# ImageNet
# Data["AUG"]["MEAN"] = [0.485, 0.456, 0.406]
# Data["AUG"]["STD"] = [0.229, 0.224, 0.225]

# miniImageNet
Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]

# ImageNet
# Data["DATA"]["IMG_SIZE"] = 224

# miniImageNet
Data["DATA"]["IMG_SIZE"] = 84

Data["MODEL"]["BACKBONE"] = 'resnet12'
# Data["MODEL"]["BACKBONE"] = 'resnet50'
# Data["MODEL"]["BACKBONE"] = 'clip'

Data["MODEL"]["PRETRAINED"] = '../pretrained_models/ce_miniImageNet_res12.ckpt'

Data["DATA"]["NUM_WORKERS"] = 8


# True for re-implementing the results in the ICML 2023 paper.
Data["AUG"]["TEST_CROP"] = True

Data["DATA"]["TEST"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 2000

# some examples of gradient-based methods. Hyperparameters need to be tuned by using search_hyperparameter.py
Data["MODEL"]["TYPE"] = "fewshot_finetune"
Data["MODEL"]["CLASSIFIER"] = "finetune"
# Data["MODEL"]["CLASSIFIER"] = "eTT"
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"fc"]# finetune_batchsize,query_feedingbatchsize,epoch,backbone_lr,classifer_lr,use_alpha,use_beta, mode
Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"fc"]# finetune
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,True,True,"NCC"]# tsa
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,True,"NCC"]# URL
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,False,False,"cc"]# CC
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,10,0.02,0.1,"eTT"]# eTT

# other adaptation classifiers
# Data["MODEL"]["TYPE"] = "Episodic_Model"
# Data["MODEL"]["CLASSIFIER"] = "LR"
# Data["MODEL"]["CLASSIFIER"] = "metaopt"
# Data["MODEL"]["CLASSIFIER"] = "proto_head"
# Data["MODEL"]["CLASSIFIER"] = "MatchingNet"

if not os.path.exists('./configs/evaluation'):
   os.makedirs('./configs/evaluation')
with open('./configs/evaluation/finetune_res12_CE.yaml', 'w') as f:
   yaml.dump(data=Data, stream=f)