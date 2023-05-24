"""
Interfaces for dataset specifications.
Adapted from original Meta-Dataset code.
"""

# coding=utf-8
# Copyright 2022 The Meta-Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import abc
import json
import os

from absl import logging
import numpy as np
import itertools
import enum
from scipy.io import loadmat
from .ImageNet_graph_operations import *
import json
import operator

# The seed is fixed, in order to ensure reproducibility of the split generation,
# exactly matching the original Meta-Dataset code.
SEED = 22

AUX_DATA_PATH = os.path.dirname(os.path.realpath(__file__))
VGGFLOWER_LABELS_PATH = f'{AUX_DATA_PATH}/VggFlower_labels.txt'
TRAFFICSIGN_LABELS_PATH = f'{AUX_DATA_PATH}/TrafficSign_labels.txt'
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class Split(enum.Enum):
  """The possible data splits."""
  TRAIN = 0
  VALID = 1
  TEST = 2


def has_file_allowed_extension(filename: str, extensions) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def gen_rand_split_inds(num_train_classes, num_valid_classes, num_test_classes, _rng):
  """Generates a random set of indices corresponding to dataset splits.
  It assumes the indices go from [0, num_classes), where the num_classes =
  num_train_classes + num_val_classes + num_test_classes. The returned indices
  are non-overlapping and cover the entire range.
  Note that in the current implementation, valid_inds and test_inds are sorted,
  but train_inds is in random order.
  Args:
    num_train_classes : int, number of (meta)-training classes.
    num_valid_classes : int, number of (meta)-valid classes.
    num_test_classes : int, number of (meta)-test classes.
    _rng : numpy fixed random number generator, used to match the split used in the 
           original benchmark.
  Returns:
    train_inds : np array of training inds.
    valid_inds : np array of valid inds.
    test_inds  : np array of test inds.
  """
  num_trainval_classes = num_train_classes + num_valid_classes
  num_classes = num_trainval_classes + num_test_classes

  # First split into trainval and test splits.
  trainval_inds = _rng.choice(
      num_classes, num_trainval_classes, replace=False)
  test_inds = np.setdiff1d(np.arange(num_classes), trainval_inds)
  # Now further split trainval into train and val.
  train_inds = _rng.choice(trainval_inds, num_train_classes, replace=False)
  valid_inds = np.setdiff1d(trainval_inds, train_inds)

  print(
      f'Created splits with {len(train_inds)} train, {len(valid_inds)} validation and {len(test_inds)} test classes.')
  return train_inds.tolist(), valid_inds.tolist(), test_inds.tolist()

def gen_sequential_split_inds(num_train_classes, num_valid_classes, num_test_classes):
  """Generates a sequential set of indices corresponding to dataset splits.
  It assumes the indices go from [0, num_classes), where the num_classes =
  num_train_classes + num_val_classes + num_test_classes. The returned indices
  are non-overlapping and cover the entire range.
  Args:
    num_train_classes : int, number of (meta)-training classes.
    num_valid_classes : int, number of (meta)-valid classes.
    num_test_classes : int, number of (meta)-test classes.
  Returns:
    train_inds : np array of training inds.
    valid_inds : np array of valid inds.
    test_inds  : np array of test inds.
  """
  train_inds = list(range(num_train_classes))
  valid_inds = list(range(num_train_classes,num_train_classes+num_valid_classes))
  test_inds = list(range(num_train_classes+num_valid_classes, num_train_classes+num_valid_classes+num_test_classes))
  return train_inds, valid_inds, test_inds

def create_spec(dataset_name, root, path_to_words=None, path_to_is_a = None, path_to_num_leaf_images = None, train_split_only = False):
  """
  create a dataset specification.
  """
  if dataset_name == "Textures":
    return create_DTD_spec(root)
  elif dataset_name == "ILSVRC":
    return create_ImageNet_spec(root, path_to_words, path_to_is_a, path_to_num_leaf_images, train_split_only)
  elif dataset_name == "Omniglot":
    return create_Omniglot_spec(root)
  elif dataset_name == "Quick Draw":
    return create_QuickDraw_spec(root)
  elif dataset_name == "Birds":
    return create_CUB_spec(root)
  elif dataset_name == "VGG Flower":
    return create_VGGFlower_spec(root)
  elif dataset_name == "Aircraft":
    return create_Aircraft_spec(root)
  elif dataset_name == "Traffic Signs":
    return create_Traffic_spec(root)
  elif dataset_name == "MSCOCO":
    return create_coco_spec(root)
  elif dataset_name == "Fungi":
    return create_fungi_spec(root)
  elif dataset_name == "MNIST":
    return create_MNIST_spec(root)
  elif dataset_name == "CIFAR10":
    return create_cifar10_spec(root)
  elif dataset_name == "CIFAR100":
    return create_cifar100_spec(root)
  elif dataset_name == "miniImageNet":
    return create_miniImageNet_spec(root)


def create_DTD_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_TRAIN_CLASSES = 33
  NUM_VALID_CLASSES = 7
  NUM_TEST_CLASSES = 7
  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)

  class_names = sorted(
        os.listdir(os.path.join(root, 'images')))

  splits = {
        Split.TRAIN: [class_names[i] for i in train_inds],
        Split.VALID: [class_names[i] for i in valid_inds],
        Split.TEST: [class_names[i] for i in test_inds]
    }


  dataset_specification = {}
  dataset_specification["name"] = "Textures"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }
  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, 'images',class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, 'images',class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification

def create_QuickDraw_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """ 
  # Sort the class names, for reproducibility.
  class_names = sorted(
        os.listdir(root))
  

  num_classes = len(class_names)
  assert num_classes == 345

  num_trainval_classes = int(0.85 * num_classes)
  num_train_classes = int(0.7 * num_classes)
  num_valid_classes = num_trainval_classes - num_train_classes
  num_test_classes = num_classes - num_trainval_classes

  _rng = np.random.RandomState(SEED)
  # Split into train, validation and test splits that have 70% / 15% / 15%
  # of the data, respectively.
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        num_train_classes, num_valid_classes, num_test_classes, _rng)




  splits = {
        Split.TRAIN: [class_names[i] for i in train_inds],
        Split.VALID: [class_names[i] for i in valid_inds],
        Split.TEST: [class_names[i] for i in test_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "Quick Draw"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }
  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))

    dataset_specification["images_per_class"][class_id].sort()

  return dataset_specification



def create_CUB_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_TRAIN_CLASSES = 140
  NUM_VALID_CLASSES = 30
  NUM_TEST_CLASSES = 30
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)

  with open(os.path.join(root, 'classes.txt'), 'r') as f:
    class_names = []
    for lines in f:
      _, class_name = lines.strip().split(' ')
      class_names.append(class_name)
  
  err_msg = 'number of classes in dataset does not match split specification'
  assert len(class_names) == NUM_TOTAL_CLASSES, err_msg

  splits = {
        Split.TRAIN: [class_names[i] for i in train_inds],
        Split.VALID: [class_names[i] for i in valid_inds],
        Split.TEST: [class_names[i] for i in test_inds]
    }


  dataset_specification = {}
  dataset_specification["name"] = "Birds"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, 'images',class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, 'images',class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification

def create_VGGFlower_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  # There are 102 classes in the VGG Flower dataset. A 70% / 15% / 15% split
  # between train, validation and test maps to roughly 71 / 15 / 16 classes,
  # respectively.
  NUM_TRAIN_CLASSES = 71
  NUM_VALID_CLASSES = 15
  NUM_TEST_CLASSES = 16
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES
  ID_LEN = 3

  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)

  # Load class names from the text file
  file_path = VGGFLOWER_LABELS_PATH
  with open(file_path) as fd:
    all_lines = fd.read()

  # First line is expected to be a comment.
  class_names = all_lines.splitlines()[1:]
  err_msg = 'number of classes in dataset does not match split specification'
  assert len(class_names) == NUM_TOTAL_CLASSES, err_msg

  # Provided class labels are numbers started at 1.
  format_str = '%%0%dd.%%s' % ID_LEN
  splits = {
      Split.TRAIN: [format_str % (i + 1, class_names[i]) for i in train_inds],
      Split.VALID: [format_str % (i + 1, class_names[i]) for i in valid_inds],
      Split.TEST: [format_str % (i + 1, class_names[i]) for i in test_inds]
  }


  imagelabels_path = os.path.join(root, 'imagelabels.mat')
  with open(imagelabels_path, 'rb') as f:
    labels = loadmat(f)['labels'][0]
  filepaths = collections.defaultdict(list)
  for i, label in enumerate(labels):
    filepaths[label].append(
        os.path.join(root, 'jpg', 'image_{:05d}.jpg'.format(i + 1)))


  dataset_specification = {}
  dataset_specification["name"] = "VGG Flower"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))


  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_label in enumerate(all_classes):
    # We encode the original ID's in the label.
    original_id = int(class_label[:ID_LEN])

    dataset_specification["id2name"][class_id] = class_label
    dataset_specification["images_per_class"][class_id] = filepaths[original_id]
  return dataset_specification

def create_Aircraft_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  # There are 100 classes in the Aircraft dataset. A 70% / 15% / 15%
  # split between train, validation and test maps to 70 / 15 / 15
  # classes, respectively.
  NUM_TRAIN_CLASSES = 70
  NUM_VALID_CLASSES = 15
  NUM_TEST_CLASSES = 15


  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)

  # Sort the class names, for reproducibility.
  class_names = sorted(
      os.listdir(root))

  assert len(class_names) == (
        NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES)

  splits = {
        Split.TRAIN: [class_names[i] for i in train_inds],
        Split.VALID: [class_names[i] for i in valid_inds],
        Split.TEST: [class_names[i] for i in test_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "Aircraft"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }
  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()


  return dataset_specification

def create_Traffic_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  # There are 43 classes in the Traffic Sign dataset, all of which are used for
  # test episodes.
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 0
  NUM_TEST_CLASSES = 43
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  # Load class names from the text file
  file_path = TRAFFICSIGN_LABELS_PATH

  with open(file_path) as fd:
    all_lines = fd.read()
  # First line is expected to be a comment.
  class_names = all_lines.splitlines()[1:]

  err_msg = 'number of classes in dataset does not match split specification'
  assert len(class_names) == NUM_TOTAL_CLASSES, err_msg

  splits = {
      'train': [],
      'valid': [],
      'test': [
          '%02d.%s' % (i, class_names[i])
          for i in range(NUM_TEST_CLASSES)
      ]
  }

  splits = {
        Split.TRAIN: [],
        Split.VALID: [],
        Split.TEST: class_names
    }

  dataset_specification = {}
  dataset_specification["name"] = "Traffic Signs"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    
    for file_ in os.listdir(os.path.join(root, '{:05d}'.format(class_id))):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, '{:05d}'.format(class_id),file_))
    dataset_specification["images_per_class"][class_id].sort()

  return dataset_specification

def create_coco_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  # There are 80 classes in the MSCOCO dataset. A 0% / 50% / 50% split
  # between train, validation and test maps to roughly 0 / 40 / 40 classes,
  # respectively.
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 40
  NUM_TEST_CLASSES = 40
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)

  annotation_path = os.path.join(root, 'instances_train2017.json')

  if not os.path.exists(annotation_path):
    raise ValueError('Annotation file %s does not exist' % annotation_path)

  with open(annotation_path, 'r') as json_file:
    annotations = json.load(json_file)
    coco_categories = annotations['categories']
    if len(coco_categories) != NUM_TOTAL_CLASSES:
      raise ValueError(
        'Total number of MSCOCO classes %d should be equal to the sum of '
        'train, val, test classes %d.' %
        (len(coco_categories), NUM_TOTAL_CLASSES))
  
  splits = {
      Split.TRAIN: [coco_categories[i]['name'] for i in train_inds],
      Split.VALID: [coco_categories[i]['name'] for i in valid_inds],
      Split.TEST: [coco_categories[i]['name'] for i in test_inds]
  }


  dataset_specification = {}
  dataset_specification["name"] = "MSCOCO"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }
  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()

  return dataset_specification

def create_fungi_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_TRAIN_CLASSES = 994
  NUM_VALID_CLASSES = 200
  NUM_TEST_CLASSES = 200

  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)
  # We ignore the original train and validation splits (the test set cannot be
  # used since it is not labeled).
  with open(os.path.join(root, 'train.json')) as f:
    original_train = json.load(f)
  with open(os.path.join(root, 'val.json')) as f:
    original_val = json.load(f)
  
  # The categories (classes) for train and validation should be the same.
  assert original_train['categories'] == original_val['categories']
  # Sort by category ID for reproducibility.
  categories = sorted(
      original_train['categories'], key=operator.itemgetter('id'))

  # Assert contiguous range [0:category_number]
  assert ([category['id'] for category in categories
          ] == list(range(len(categories))))

  # Some categories share the same name (see
  # https://github.com/visipedia/fgvcx_fungi_comp/issues/1)
  # so we include the category id in the label.
  labels = [
      '{:04d}.{}'.format(category['id'], category['name'])
      for category in categories
  ]


  splits = {
        Split.TRAIN: [labels[i] for i in train_inds],
        Split.VALID: [labels[i] for i in valid_inds],
        Split.TEST: [labels[i] for i in test_inds]
    }
  
  image_list = original_train['images'] + original_val['images']
  image_id_dict = {}
  for image in image_list:
    # assert this image_id was not previously added
    assert image['id'] not in image_id_dict
    image_id_dict[image['id']] = image

  # Add a class annotation to every image in image_id_dict.
  annotations = original_train['annotations'] + original_val['annotations']
  for annotation in annotations:
    # assert this images_id was not previously annotated
    assert 'class' not in image_id_dict[annotation['image_id']]
    image_id_dict[annotation['image_id']]['class'] = annotation['category_id']

  # dict where the class is the key.
  class_filepaths = collections.defaultdict(list)
  for image in image_list:
    class_filepaths[image['class']].append(
        os.path.join(root, image['file_name']))

  dataset_specification = {}
  dataset_specification["name"] = "Fungi"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}

  for class_id, class_label in enumerate(all_classes):
    # Extract the "category_id" information from the class label
    category_id = int(class_label[:4])
    # Check that the key is actually in `class_filepaths`, so that an empty
    # list is not accidentally used.
    if category_id not in class_filepaths:
      raise ValueError('class_filepaths does not contain paths to any '
                        'image for category %d. Existing categories are: %s.' %
                        (category_id, class_filepaths.keys()))
    class_paths = class_filepaths[category_id]
    dataset_specification["id2name"][class_id] = class_label
    dataset_specification["images_per_class"][class_id] = class_paths
  return dataset_specification

def create_Omniglot_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    superclasses_per_split: number of superclasses per split.
    classes_per_superclass: number of classes per superclass.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  # We chose the 5 smallest alphabets (i.e. those with the least characters)
  # out of the 'background' set of alphabets that are intended for train/val
  # We keep the 'evaluation' set of alphabets for testing exclusively
  # The chosen alphabets have 14, 14, 16, 17, and 20 characters, respectively.
  validation_alphabets = [
      'Blackfoot_(Canadian_Aboriginal_Syllabics)',
      'Ojibwe_(Canadian_Aboriginal_Syllabics)',
      'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Tagalog',
      'Alphabet_of_the_Magi'
  ]

  training_alphabets = []
  data_path_trainval = os.path.join(root, 'images_background')
  for alphabet_name in sorted(os.listdir(data_path_trainval)):
      if alphabet_name not in validation_alphabets:
        training_alphabets.append(alphabet_name)
  assert len(training_alphabets) + len(validation_alphabets) == 30

  data_path_test = os.path.join(root, 'images_evaluation')
  test_alphabets = sorted(os.listdir(data_path_test))
  assert len(test_alphabets) == 20

  class_names = {}
  images_per_class = {}
  superclass_names = {}
  classes_per_superclass = collections.defaultdict(int)
  superclasses_per_split = {
        Split.TRAIN: 0,
        Split.VALID: 0,
        Split.TEST: 0
    }
  images_per_class = {}
  def parse_split_data(split, alphabets, alphabets_path):
    # Each alphabet is a superclass.
    for alphabet_folder_name in alphabets:
      alphabet_path = os.path.join(alphabets_path, alphabet_folder_name)
      # Each character is a class.
      for char_folder_name in sorted(os.listdir(alphabet_path)):
        class_path = os.path.join(alphabet_path, char_folder_name)
        class_label = len(class_names)

        class_names[class_label] = '{}-{}'.format(alphabet_folder_name,
                                                       char_folder_name)
        images_per_class[class_label] = []
        for file_ in os.listdir(class_path):
          if is_image_file(file_):
            images_per_class[class_label].append(os.path.join(class_path,file_))
        images_per_class[class_label].sort()
        # Add this character to the count of subclasses of this superclass.
        superclass_label = len(superclass_names)
        classes_per_superclass[superclass_label] += 1

      # Add this alphabet as a superclass.
      superclasses_per_split[split] += 1
      superclass_names[superclass_label] = alphabet_folder_name
  

  parse_split_data(Split.TRAIN, training_alphabets,
                          data_path_trainval)
  parse_split_data(Split.VALID, validation_alphabets,
                        data_path_trainval)
  parse_split_data(Split.TEST, test_alphabets,
                        data_path_test)
                        

  dataset_specification = {}
  dataset_specification["name"] = "Omniglot"
  dataset_specification["superclasses_per_split"] = superclasses_per_split
  dataset_specification["id2name"] = class_names
  dataset_specification["classes_per_superclass"] = classes_per_superclass

  dataset_specification["images_per_class"] = images_per_class

  return dataset_specification

  

def create_ImageNet_spec(root, 
                         path_to_words = None, 
                         path_to_is_a = None, 
                         path_to_num_leaf_images=None,
                         train_split_only = False
                         ):
  """
  Args:
    root: path to ImageNet training set.
    path_to_words: path to words.txt.
    path_to_is_a: path to wordnet.is_a.txt
    path_to_num_leaf_images: path to save computed dict mapping the WordNet id 
                             of each ILSVRC 2012 class to its number of images.
    train_split_only: whether use all classes of ImageNet for training.

  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
    split_subgraph: a dict mapping each split to the set of Synsets in the
                    subgraph of that split.
    class_names_to_ids: a dictionary mapping real name to class id.

  """
  # Load lists of image names that are duplicates with images in other
  # datasets. They will be skipped from ImageNet.
  files_to_skip = set()
  for other_dataset in ('Caltech101', 'Caltech256', 'CUBirds'):
    duplicates_file = os.path.join(AUX_DATA_PATH, 'ImageNet_{}_duplicates.txt'.format(other_dataset))

    with open(duplicates_file) as fd:
        duplicates = fd.read()
    lines = duplicates.splitlines()
    for l in lines:
        # Skip comment lines
        l = l.strip()
        if l.startswith('#'):
          continue
        # Lines look like:
        # 'synset/synset_imgnumber.JPEG  # original_file_name.jpg\n'.
        # Extract only the 'synset_imgnumber.JPG' part.
        file_path = l.split('#')[0].strip()
        file_name = os.path.basename(file_path)
        files_to_skip.add(file_name)

  synsets = {}
  if not path_to_words:
    path_to_words = os.path.join(root, 'words.txt')

  with open(path_to_words) as f:
    for line in f:
      wn_id, words = line.rstrip().split('\t')
      synsets[wn_id] = Synset(wn_id, words, set(), set())
  
  # Populate the parents / children arrays of these Synsets.
  if not path_to_is_a:
    path_to_is_a = os.path.join(root, 'wordnet.is_a.txt')

  with open(path_to_is_a, 'r') as f:
    for line in f:
      parent, child = line.rstrip().split(' ')
      synsets[parent].children.add(synsets[child])
      synsets[child].parents.add(synsets[parent])

  wn_ids_2012 = os.listdir(root)
  wn_ids_2012 = set(
    entry for entry in wn_ids_2012
    if os.path.isdir(os.path.join(root, entry)))

  # all leaves in ImageNet
  synsets_2012 = [s for s in synsets.values() if s.wn_id in wn_ids_2012]
  assert len(wn_ids_2012) == len(synsets_2012)

  # Get a dict mapping each WordNet id of ILSVRC 2012 to its number of images.
  num_synset_2012_images = get_num_synset_2012_images(root, path_to_num_leaf_images,
                                                      synsets_2012,
                                                      files_to_skip)

  # Get the graph of all and only the ancestors of the ILSVRC 2012 classes.
  sampling_graph = create_sampling_graph(synsets_2012)

  # Create a dict mapping each node to its reachable leaves.
  spanning_leaves = get_spanning_leaves(sampling_graph)

  # Create a dict mapping each node in sampling graph to the number of images of
  # ILSVRC 2012 synsets that live in the sub-graph rooted at that node.
  num_images = get_num_spanning_images(spanning_leaves, num_synset_2012_images)

  if train_split_only:
    # We are keeping all graph for training.
    valid_test_roots = None
    splits = {
        Split.TRAIN: spanning_leaves,
        Split.VALID: set(),
        Split.TEST: set()
    }
  else:
    # Create class splits, each with its own sampling graph.
    # Choose roots for the validation and test subtrees (see the docstring of
    # create_splits for more information on how these are used).
    valid_test_roots = {
        'valid': get_synset_by_wnid('n02075296', sampling_graph),  # 'carnivore'
        'test':
            get_synset_by_wnid('n03183080', sampling_graph)  # 'device'
    }
    # The valid_test_roots returned here correspond to the same Synsets as in
    # the above dict, but are the copied versions of them for each subgraph.
    splits, valid_test_roots = create_splits(
        spanning_leaves, Split, valid_test_roots=valid_test_roots)


  # Compute num_images for each split.
  split_num_images = {}
  split_num_images[Split.TRAIN] = get_num_spanning_images(
      get_spanning_leaves(splits[Split.TRAIN]), num_synset_2012_images)
  split_num_images[Split.VALID] = get_num_spanning_images(
      get_spanning_leaves(splits[Split.VALID]), num_synset_2012_images)
  split_num_images[Split.TEST] = get_num_spanning_images(
      get_spanning_leaves(splits[Split.TEST]), num_synset_2012_images)


  # Get a list of synset id's assigned to each split.
  def _get_synset_ids(split):
    """Returns a list of synset id's of the classes assigned to split."""
    return sorted([
        synset.wn_id for synset in get_leaves(
            splits[split])
    ])
  train_synset_ids = _get_synset_ids(Split.TRAIN)
  valid_synset_ids = _get_synset_ids(Split.VALID)
  test_synset_ids = _get_synset_ids(Split.TEST)

  all_synset_ids = train_synset_ids + valid_synset_ids + test_synset_ids

  # By construction of all_synset_ids, we are guaranteed to get train synsets
  # before validation synsets, and validation synsets before test synsets.
  # Therefore the assigned class_labels will respect that partial order.
  class_names = {}
  for class_label, synset_id in enumerate(all_synset_ids):
      class_names[class_label] = synset_id

  dataset_specification = {}
  dataset_specification["name"] = "ILSVRC"
  dataset_specification["split_subgraph"] = splits
  dataset_specification["class_names_to_ids"] = dict(
    zip(class_names.values(), class_names.keys()))
  dataset_specification["id2name"] = class_names
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(train_synset_ids),
        Split.VALID: len(valid_synset_ids),
        Split.TEST: len(test_synset_ids)
    }

  
  dataset_specification["images_per_class"] = {}


  for class_id, class_name in class_names.items():
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification

def create_MNIST_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 0
  NUM_TEST_CLASSES = 10
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))

  assert len(class_names) == 10

  splits = {
        Split.TRAIN: [class_names[i] for i in train_inds],
        Split.VALID: [class_names[i] for i in valid_inds],
        Split.TEST: [class_names[i] for i in test_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "MNIST"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification

def create_cifar10_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 0
  NUM_TEST_CLASSES = 10
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))
  
  assert len(class_names) == 10

  splits = {
        Split.TRAIN: [class_names[i] for i in train_inds],
        Split.VALID: [class_names[i] for i in valid_inds],
        Split.TEST: [class_names[i] for i in test_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "CIFAR10"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification

def create_cifar100_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_TRAIN_CLASSES = 0
  NUM_VALID_CLASSES = 0
  NUM_TEST_CLASSES = 100
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  _rng = np.random.RandomState(SEED)
  train_inds, valid_inds, test_inds = gen_rand_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES, _rng)

  class_names = sorted(
        os.listdir(root))
  assert len(class_names) == 100

  splits = {
        Split.TRAIN: [class_names[i] for i in train_inds],
        Split.VALID: [class_names[i] for i in valid_inds],
        Split.TEST: [class_names[i] for i in test_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "CIFAR100"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []
    for file_ in os.listdir(os.path.join(root, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification

def create_miniImageNet_spec(root):
  """
  Return a dataset specification that includes:
    name: The name of the dataset.
    num_classes_per_split: number of images per split.
    images_per_class: a dictionary containing all paths of images in each class.
    id2name: a dictionary mapping class id to real name.
  """
  NUM_TRAIN_CLASSES = 64
  NUM_VALID_CLASSES = 16
  NUM_TEST_CLASSES = 20
  NUM_TOTAL_CLASSES = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + NUM_TEST_CLASSES

  train_inds, valid_inds, test_inds = gen_sequential_split_inds(
        NUM_TRAIN_CLASSES, NUM_VALID_CLASSES, NUM_TEST_CLASSES)

  train_class_names = sorted(
        os.listdir(os.path.join(root,"train")))
  valid_class_names = sorted(
        os.listdir(os.path.join(root,"val")))
  test_class_names = sorted(
        os.listdir(os.path.join(root,"test")))
  class_names = train_class_names+valid_class_names+test_class_names

  assert len(train_class_names) == 64 and len(valid_class_names) == 16 and len(test_class_names) == 20

  splits = {
        Split.TRAIN: [class_names[i] for i in train_inds],
        Split.VALID: [class_names[i] for i in valid_inds],
        Split.TEST: [class_names[i] for i in test_inds]
    }

  dataset_specification = {}
  dataset_specification["name"] = "miniImageNet"
  dataset_specification["num_classes_per_split"] = {
        Split.TRAIN: len(splits[Split.TRAIN]),
        Split.VALID: len(splits[Split.VALID]),
        Split.TEST: len(splits[Split.TEST])
    }

  all_classes = list(
        itertools.chain(splits[Split.TRAIN], splits[Split.VALID], splits[Split.TEST]))
  class_names = {}

  dataset_specification["images_per_class"] = {}

  dataset_specification["id2name"] = {}


  for class_id, class_name in enumerate(all_classes):
    dataset_specification["id2name"][class_id] = class_name
    dataset_specification["images_per_class"][class_id] = []

    if class_id <64:
      split = "train"
    elif class_id <80:
      split = "val"
    else:
      split = "test"
    for file_ in os.listdir(os.path.join(root, split, class_name)):
      if is_image_file(file_):
        dataset_specification["images_per_class"][class_id].append(os.path.join(root, split, class_name,file_))
    dataset_specification["images_per_class"][class_id].sort()
  return dataset_specification





def get_bilevel_classes(split, dataset_spec):
  """Gets the sequence of class labels for a split.
    Labels are returned ordered and without gaps.
    Args:
      split: A Split, the split for which to get classes.
    Returns:
      The sequence of classes for the split.
    """
  if split == Split.TRAIN:
    offset = 0
  elif split == Split.VALID:
    previous_superclasses = range(
        0, dataset_spec["superclasses_per_split"][Split.TRAIN])
    offset = sum([
        dataset_spec["classes_per_superclass"][superclass_id]
        for superclass_id in previous_superclasses
    ])
  elif split == Split.TEST:
    previous_superclasses = range(
        0, dataset_spec["superclasses_per_split"][Split.TRAIN] +
        dataset_spec["superclasses_per_split"][Split.VALID])
    offset = sum([
        dataset_spec["classes_per_superclass"][superclass_id]
        for superclass_id in previous_superclasses
    ])
  else:
    raise ValueError('Invalid dataset split.')

  num_split_classes = sum([
      dataset_spec["classes_per_superclass"][superclass_id]
      for superclass_id in get_classes(split, dataset_spec["superclasses_per_split"])
  ])
  return range(offset, offset + num_split_classes)


def get_classes(split, classes_per_split):
  """Gets the sequence of class labels for a split.
  Class id's are returned ordered and without gaps.
  Args:
    split: A Split, the split for which to get classes.
    classes_per_split: Matches each Split to the number of its classes.
  Returns:
    The sequence of classes for the split.
  Raises:
    ValueError: An invalid split was specified.
  """

  num_classes = classes_per_split[split]

  # Find the starting index of classes for the given split.
  if split == Split.TRAIN:
    offset = 0
  elif split == Split.VALID:
    offset = classes_per_split[Split.TRAIN]
  elif split == Split.TEST:
    offset = (
        classes_per_split[Split.TRAIN] +
        classes_per_split[Split.VALID])
  else:
    raise ValueError('Invalid dataset split.')

  # Get a contiguous range of classes from split.
  return range(offset, offset + num_classes)




def get_total_images_per_class(data_spec, class_id=None, pool=None):
  """Returns the total number of images of a class in a data_spec and pool.
  Args:
    data_spec: A DatasetSpecification, or BiLevelDatasetSpecification.
    class_id: The class whose number of images will be returned. If this is
      None, it is assumed that the dataset has the same number of images for
      each class.
    pool: A string ('train' or 'test', optional) indicating which example-level
      split to select, if the current dataset has them.
  Raises:
    ValueError: when
      - no class_id specified and yet there is class imbalance, or
      - no pool specified when there are example-level splits, or
      - pool is specified but there are no example-level splits, or
      - incorrect value for pool.
    RuntimeError: the DatasetSpecification is out of date (missing info).
  """
  if class_id is None:
    if len(set(data_spec.images_per_class.values())) != 1:
      raise ValueError('Not specifying class_id is okay only when all classes'
                       ' have the same number of images')
    class_id = 0

  if class_id not in data_spec.images_per_class:
    raise RuntimeError('The DatasetSpecification should be regenerated, as '
                       'it does not have a non-default value for class_id {} '
                       'in images_per_class.'.format(class_id))
  num_images = data_spec.images_per_class[class_id]

  if pool is None:
    if isinstance(num_images, abc.Mapping):
      raise ValueError('DatasetSpecification {} has example-level splits, so '
                       'the "pool" argument has to be set (to "train" or '
                       '"test".'.format(data_spec.name))
  elif not data.POOL_SUPPORTED:
    raise NotImplementedError('Example-level splits or pools not supported.')

  return num_images


