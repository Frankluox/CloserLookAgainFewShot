import yaml
import os

Data = {}
Data["DATA"] = {}

Data["DATA"]["PATH_TO_WORDS"] = "/home/luoxu/data/words.txt"
Data["DATA"]["PATH_TO_IS_A"] = "/home/luoxu/data/wordnet.is_a.txt"
Data["DATA"]["PATH_TO_NUM_LEAF_IMAGES"] = "data/ImageNet_num_images_perclass.json"

Data["DATA"]["TRAIN"] = {}

Data["DATA"]["TRAIN"]["DATASET_ROOTS"] = ["/home/wuhao/data/mini_imagenet/images_imagefolder"]
Data["DATA"]["TRAIN"]["DATASET_NAMES"] = ["miniImageNet"]

# Data["DATA"]["TRAIN"]["DATASET_ROOTS"] = ["PATH-TO-miniIMAGENET"]
# Data["DATA"]["TRAIN"]["DATASET_NAMES"] = ["miniImageNet"]

Data["DATA"]["TRAIN"]["IS_EPISODIC"] = True

Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"] = {}

Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 5
Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]
Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False



Data["DATA"]["VALID"] = {}





Data["DATA"]["VALID"]["DATASET_ROOTS"] = ["/home/wuhao/data/mini_imagenet/images_imagefolder"]
Data["DATA"]["VALID"]["DATASET_NAMES"] = ["miniImageNet"]

# Data["DATA"]["TRAIN"]["DATASET_ROOTS"] = ["PATH-TO-miniIMAGENET"]
# Data["DATA"]["TRAIN"]["DATASET_NAMES"] = ["miniImageNet"]



Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"] = {}
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_WAYS"] = 5
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"] = 5
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"] = 15
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["MAX_NUM_QUERY"] = 15
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["USE_DAG_HIERARCHY"] = False
Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["MIN_EXAMPLES_IN_CLASS"] = Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_SUPPORT"]+Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_QUERY"]

Data["DATA"]["VALID"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 600
Data["DATA"]["VALID"]["BATCH_SIZE"] = 8


Data["AUG"] = {}
Data["AUG"]["MEAN"] = [0.4712, 0.4499, 0.4031]
Data["AUG"]["STD"] = [0.2726, 0.2634, 0.2794]

Data["OUTPUT"] = "../new_metadataset_result"

Data["MODEL"] = {}
Data["MODEL"]["TYPE"] = "Episodic_Model"
Data["MODEL"]["CLASSIFIER"] = "proto_head"
Data["MODEL"]["NAME"] = "miniImageNet_Res12_PN"


Data["MODEL"]["BACKBONE"] = 'resnet12'

Data["DATA"]["IMG_SIZE"] = 84
Data["DATA"]["NUM_WORKERS"] = 8
Data["GPU_ID"] = 2
Data["TRAIN"] = {}
Data["TRAIN"]["EPOCHS"] = 60

Data["DATA"]["TRAIN"]["BATCH_SIZE"] = 2

Data["TRAIN"]["BASE_LR"] = 0.025*Data["DATA"]["TRAIN"]["BATCH_SIZE"]


Data["DATA"]["TRAIN"]["EPISODE_DESCR_CONFIG"]["NUM_TASKS_PER_EPOCH"] = 1000



if not os.path.exists('./configs/PN'):
   os.makedirs('./configs/PN')

with open('./configs/PN/miniImageNet_res12.yaml', 'w') as f:
   yaml.dump(data=Data, stream=f)