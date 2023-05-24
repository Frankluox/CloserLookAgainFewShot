from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from .dataset_spec import Split, create_spec
from .bulid_transforms import build_Torch_transform
import torch
import numpy as np
from .sampling import EpisodeSampler, BatchSampler
import collections

def find_index(random_number, sampling_frequency):
    """
    determine dataset index given sampling frequencies.
    a random float number in [0,1] is used to sample the dataset index
    according to the frequencies.

    args:
        random_number: a float number in [0,1]
        sampling_frequency: a list of float numbers summing to 1.
    """
    index = 0
    sum_ = sampling_frequency[0]
    while sum_ < random_number:
        index += 1
        sum_ += sampling_frequency[index]
    return index

class TorchDataset(Dataset):
    def __init__(self,
                 split,
                 dataset_names, 
                 dataset_roots,
                 sampling_frequency,
                 batch_size, 
                 seed,
                 transforms,
                 is_episodic=True, 
                 episode_descr_config = None,
                 iteration_per_epoch = None,
                 shuffle = False,
                 path_to_words = None, 
                 path_to_is_a = None, 
                 path_to_num_leaf_images=None,
                 train_split_only = False
                 ):
        """
        split: which split to sample from
        dataset_names: a list of all dataset names that will be used
        dataset_roots: a list of all dataset roots
        sampling_frequency: a list of float numbers representing sampling
                            frequencies of each dataset.
        batch_size: number of tasks per iteration(for episodic training/test)
                    or batch size
        seed: seed for sampling
        transforms: image transformations
        is_episodic: episodic training/test or not
        episode_descr_config: detailed configurations about how to sample a episode
        iteration_per_epoch: the number of iterations per epoch. Only works for non-episodic training/test.
        shuffle: shuffle a batch or not. Only works for non-episodic training/test.
        path_to_words: path to words.txt.
        path_to_is_a: path to wordnet.is_a.txt
        path_to_num_leaf_images: path to save computed dict mapping the WordNet id 
                                of each ILSVRC 2012 class to its number of images.
        train_split_only: whether use all classes of ImageNet for training.
        """
        
        self._rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.sampling_frequency = sampling_frequency
        self.is_episodic = is_episodic
        self.shuffle = shuffle
        self.iteration_per_epoch = iteration_per_epoch
        self.episode_descr_config = episode_descr_config
        self.samplers = [] 
        self.transforms = transforms
        self.seed = seed
        
        assert len(dataset_names)==len(sampling_frequency)==len(dataset_roots)
        
        # sum of frequencies
        sum_frequency = 0

        for fre in sampling_frequency:
            sum_frequency += fre

        # make sure that the sum of frequencies is no less than and very close to 1.
        assert abs(sum_frequency-1) < 0.01 and sum_frequency >= 1

        # construct dataset specifications
        dataset_specs = []
        for i, dataset_name in enumerate(dataset_names):
            if dataset_name == "ILSVRC":
                dataset_specs.append(create_spec(dataset_name,
                                                 dataset_roots[i],
                                                 path_to_words, 
                                                 path_to_is_a, 
                                                 path_to_num_leaf_images,
                                                 train_split_only))
            else:
                dataset_specs.append(create_spec(dataset_name, dataset_roots[i]))

        # construct samplers of each dataset
        if is_episodic:
            assert episode_descr_config is not None
            for dataset_spec in dataset_specs:
                self.samplers.append(EpisodeSampler(
                    seed,dataset_spec,split,episode_descr_config))
            self.iteration_per_epoch = episode_descr_config.NUM_TASKS_PER_EPOCH//self.batch_size
        else:
            self.num_classes = []
            for dataset_spec in dataset_specs:
                self.samplers.append(BatchSampler(seed, dataset_spec,split))
                self.num_classes.append(len(self.samplers[-1].class_set))

            if iteration_per_epoch is None:
                # determine iterations by the length of dataset
                assert len(dataset_names)==1
                self.iteration_per_epoch = (self.samplers[0].length+batch_size-1)//batch_size

        # all task/batches of an epoch
        self.all_tasks = []



    def __len__(self):
        return self.iteration_per_epoch

    def set_epoch(self) -> None:
        # sample all tasks/batches of images of an epoch before the epoch starts
        self.all_tasks = []

        # for one-dataset batch sampling, the dataset sampler should be reset.
        if not self.is_episodic and len(self.sampling_frequency)==1:
            self.samplers[0].init()
        
        # sample tasks/batches
        for _ in range(self.iteration_per_epoch):
            # [0,1] uniform sampling to determine the dataset to sample from
            random_number = self._rng.uniform()

            # dataset index
            index = find_index(random_number, self.sampling_frequency)

            if self.is_episodic:
                self.all_tasks.append([index, *self.samplers[index].sample_multiple_episode(self.batch_size, self.episode_descr_config.SEQUENTIAL_SAMPLING)])
            else:
                self.all_tasks.append([index, *self.samplers[index].sample_batch(self.batch_size)])

    def __getitem__(self, index):
        # sanple a task/batch

        dataset_index, image_paths, labels = self.all_tasks[index]
        images = []
        if self.is_episodic:
            for task in image_paths:
                processed_task = collections.defaultdict(list)
                for path in task["support"]:
                    image = Image.open(path).convert('RGB')
                    processed_task["support"].append(self.transforms(image))
                for path in task["query"]:
                    image = Image.open(path).convert('RGB')
                    processed_task["query"].append(self.transforms(image))
                processed_task["support"] = torch.stack(processed_task["support"])
                processed_task["query"] = torch.stack(processed_task["query"])
                images.append(processed_task)
        else:
            for path in image_paths:
                image = Image.open(path).convert('RGB')
                images.append(self.transforms(image))
            images = torch.stack(images)
        
        return dataset_index, images, labels

    




def create_torch_dataloader(split, config):
    # create a dataloader
    is_train = False
    if split == Split.TRAIN:
        config_ = config.DATA.TRAIN
        is_train = True
    elif split == Split.VALID:
        config_ = config.DATA.VALID
    elif split == Split.TEST:
        config_ = config.DATA.TEST
    else:
        raise ValueError("Split name does not exist.")

    transforms = build_Torch_transform(is_train, config)

    path_to_words = config.DATA.PATH_TO_WORDS
    path_to_is_a = config.DATA.PATH_TO_IS_A
    path_to_num_leaf_images = config.DATA.PATH_TO_NUM_LEAF_IMAGES



    dataset = TorchDataset(split,    
                            config_.DATASET_NAMES,
                            config_.DATASET_ROOTS,
                            config_.SAMPLING_FREQUENCY,
                            config_.BATCH_SIZE, 
                            config.SEED,
                            transforms,
                            config_.IS_EPISODIC,
                            config_.EPISODE_DESCR_CONFIG,
                            config_.ITERATION_PER_EPOCH,
                            config_.SHUFFLE,
                            config.DATA.PATH_TO_WORDS,
                            config.DATA.PATH_TO_IS_A,
                            config.DATA.PATH_TO_NUM_LEAF_IMAGES,
                            config.DATA.TRAIN_SPLIT_ONLY
                            )
        
    loader = DataLoader(dataset,
                        num_workers = config.DATA.NUM_WORKERS,
                        pin_memory = config.DATA.PIN_MEMORY)
    
    return loader, dataset

    
    
    


