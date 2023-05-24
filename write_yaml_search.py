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


Data["DATA"]["VALID"] = {}


Data["DATA"]["VALID"]["DATASET_ROOTS"] = [roots[2]]
Data["DATA"]["VALID"]["DATASET_NAMES"] = [names[2]]


Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"] = {}
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 5
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]
Data["DATA"]["VALID"]["BATCH_SIZE"] = 8
Data["OUTPUT"] = "../new_metadataset_result"
Data["MODEL"] = {}

Data["MODEL"]["NAME"] = "evaluation"
Data["GPU_ID"] = 2

# 1 if use sequential sampling in the original false Meta-Dataset sampling
# 1 used to re-implement the results in the ICML 2023 paper; 0, however, is recommended
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["SEQUENTIAL_SAMPLING"] = 1

Data["AUG"] = {}

# ImageNet
# Data["AUG"]["MEAN"] = [0.485, 0.456, 0.406]
# Data["AUG"]["STD"] = [0.229, 0.224, 0.225]

# miniImageNet
Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]


Data["DATA"]["IMG_SIZE"] = 84

Data["MODEL"]["BACKBONE"] = 'resnet12'
Data["MODEL"]["PRETRAINED"] = '../pretrained_models/ce_miniImageNet_res12.ckpt'

Data["DATA"]["NUM_WORKERS"] = 8

Data["AUG"]["TEST_CROP"] = True

Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 50


# some examples of gradient-based methods. Hyperparameters need to be tuned by using search_hyperparameter.py
Data["MODEL"]["TYPE"] = "fewshot_finetune"
Data["MODEL"]["CLASSIFIER"] = "finetune"
# Data["MODEL"]["CLASSIFIER"] = "eTT"

# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,100,0.02,0.1,False,False,"fc"]# finetune_batchsize,query_feedingbatchsize,epoch,backbone_lr,classifer_lr,use_alpha,use_beta, mode
Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,False,False,"fc"]# finetune
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,True,True,"NCC"]# tsa
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,False,True,"NCC"]# URL
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,False,False,"cc"]# CC
# Data["MODEL"]["CLASSIFIER_PARAMETERS"] = [100,100,None,None,None,"eTT"]# eTT



Data["SEARCH_HYPERPARAMETERS"] = {}

# change this
Data["SEARCH_HYPERPARAMETERS"]["LR_BACKBONE_RANGE"] = [0.01,0.05,0.25]
Data["SEARCH_HYPERPARAMETERS"]["LR_HEAD_RANGE"] = [0.02,0.1,0.5]
Data["SEARCH_HYPERPARAMETERS"]["EPOCH_RANGE"] = [10,20,30]

if not os.path.exists('./configs/search'):
   os.makedirs('./configs/search')
with open('./configs/search/finetune_res12_CE.yaml', 'w') as f:
   yaml.dump(data=Data, stream=f)