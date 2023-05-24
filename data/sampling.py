"""
Data sampling for both episodic and non-episodic training/testing.
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

from .dataset_spec import get_classes,create_spec, Split, get_bilevel_classes
import numpy as np
from .ImageNet_graph_operations import get_leaves, get_spanning_leaves
import torch
import logging
import collections
MAX_SPANNING_LEAVES_ELIGIBLE = 392





def sample_num_ways_uniformly(num_classes, min_ways, max_ways, rng=None):
    """Samples a number of ways for an episode uniformly and at random.
    The support of the distribution is [min_ways, num_classes], or
    [min_ways, max_ways] if num_classes > max_ways.
    Args:
    num_classes: int, number of classes.
    min_ways: int, minimum number of ways.
    max_ways: int, maximum number of ways. Only used if num_classes > max_ways.
    rng: np.random.RandomState used for sampling.
    Returns:
    num_ways: int, number of ways for the episode.
    """
    rng = rng or RNG
    max_ways = min(max_ways, num_classes)
    sample_ways = rng.randint(low=min_ways, high=max_ways + 1)
    return sample_ways


def sample_class_ids_uniformly(num_ways, rel_classes, rng=None):
    """Samples the (relative) class IDs for the episode.
    Args:
    num_ways: int, number of ways for the episode.
    rel_classes: list of int, available class IDs to sample from.
    rng: np.random.RandomState used for sampling.
    Returns:
    class_ids: np.array, class IDs for the episode, with values in rel_classes.
    """
    rng = rng or RNG
    return rng.choice(rel_classes, num_ways, replace=False)


def compute_num_query(images_per_class, max_num_query, num_support):
  """Computes the number of query examples per class in the episode.
  Query sets are balanced, i.e., contain the same number of examples for each
  class in the episode.
  The number of query examples satisfies the following conditions:
  - it is no greater than `max_num_query`
  - if support size is unspecified, it is at most half the size of the
    smallest class in the episode
  - if support size is specified, it is at most the size of the smallest class
    in the episode minus the max support size.
  Args:
    images_per_class: np.array, number of images for each class.
    max_num_query: int, number of images for each class.
    num_support: int or tuple(int, int), number (or range) of support
      images per class.
  Returns:
    num_query: int, number of query examples per class in the episode.
  """
  if num_support is None:
    if images_per_class.min() < 2:
      raise ValueError('Expected at least 2 images per class.')
    return np.minimum(max_num_query, (images_per_class // 2).min())
  elif isinstance(num_support, int):
    max_support = num_support
  else:
    _, max_support = num_support
  if (images_per_class - max_support).min() < 1:
    raise ValueError(
        'Expected at least {} images per class'.format(max_support + 1))
  return np.minimum(max_num_query, images_per_class.min() - max_support)


def sample_support_set_size(num_remaining_per_class,
                            max_support_size_contrib_per_class,
                            max_support_set_size,
                            rng=None):
  """Samples the size of the support set in the episode.
  That number is such that:
  * The contribution of each class to the number is no greater than
    `max_support_size_contrib_per_class`.
  * It is no greater than `max_support_set_size`.
  * The support set size is greater than or equal to the number of ways.
  Args:
    num_remaining_per_class: np.array, number of images available for each class
      after taking into account the number of query images.
    max_support_size_contrib_per_class: int, maximum contribution for any given
      class to the support set size. Note that this is not a limit on the number
      of examples of that class in the support set; this is a limit on its
      contribution to computing the support set _size_.
    max_support_set_size: int, maximum size of the support set.
    rng: np.random.RandomState used for sampling.
  Returns:
    support_set_size: int, size of the support set in the episode.
  """
  rng = rng or RNG
  if max_support_set_size < len(num_remaining_per_class):
    raise ValueError('max_support_set_size is too small to have at least one '
                     'support example per class.')
  beta = rng.uniform()
  # print('hh')
  # print(beta)
  

  support_size_contributions = np.minimum(max_support_size_contrib_per_class,
                                          num_remaining_per_class)
  # print(support_size_contributions)
  # print(np.floor(beta * support_size_contributions + 1).sum())
  return np.minimum(
      # Taking the floor and adding one is equivalent to sampling beta uniformly
      # in the (0, 1] interval and taking the ceiling of its product with
      # `support_size_contributions`. This ensures that the support set size is
      # at least as big as the number of ways.
      np.floor(beta * support_size_contributions + 1).sum(),
      max_support_set_size)


def sample_num_support_per_class(images_per_class,
                                 num_remaining_per_class,
                                 support_set_size,
                                 min_log_weight,
                                 max_log_weight,
                                 rng=None):
  """Samples the number of support examples per class.
  At a high level, we wish the composition to loosely match class frequencies.
  Sampling is done such that:
  * The number of support examples per class is no greater than
    `support_set_size`.
  * The number of support examples per class is no greater than the number of
    remaining examples per class after the query set has been taken into
    account.
  Args:
    images_per_class: np.array, number of images for each class.
    num_remaining_per_class: np.array, number of images available for each class
      after taking into account the number of query images.
    support_set_size: int, size of the support set in the episode.
    min_log_weight: float, minimum log-weight to give to any particular class.
    max_log_weight: float, maximum log-weight to give to any particular class.
    rng: np.random.RandomState used for sampling.
  Returns:
    num_support_per_class: np.array, number of support examples for each class.
  """
  rng = rng or RNG
  if support_set_size < len(num_remaining_per_class):
    raise ValueError('Requesting smaller support set than the number of ways.')
  if np.min(num_remaining_per_class) < 1:
    raise ValueError('Some classes have no remaining examples.')

  # Remaining number of support examples to sample after we guarantee one
  # support example per class.
  remaining_support_set_size = support_set_size - len(num_remaining_per_class)

  unnormalized_proportions = images_per_class * np.exp(
      rng.uniform(min_log_weight, max_log_weight, size=images_per_class.shape))
  support_set_proportions = (
      unnormalized_proportions / unnormalized_proportions.sum())

  # This guarantees that there is at least one support example per class.
  num_desired_per_class = np.floor(
      support_set_proportions * remaining_support_set_size).astype('int32') + 1

  return np.minimum(num_desired_per_class, num_remaining_per_class)




class EpisodeSampler(object):
  """Generates samples of an episode.
  In particular, for each episode, it will sample all files and labels of a task.
  """

  def __init__(self,
               seed,
               dataset_spec,
               split,
               episode_descr_config,
               ):
    """
    seed: seed for sampling
    dataset_spec: dataset specification
    split: which split to sample from
    episode_descr_config: detailed configurations about how to sample a episode
    """

    # Fixing seed for sampling
    self._rng = np.random.RandomState(seed)


    self.dataset_spec = dataset_spec
    self.split = split

    self.num_ways = episode_descr_config.NUM_WAYS
    self.num_support = episode_descr_config.NUM_SUPPORT
    self.num_query = episode_descr_config.NUM_QUERY
    self.min_ways = episode_descr_config.MIN_WAYS
    self.max_ways_upper_bound = episode_descr_config.MAX_WAYS_UPPER_BOUND
    self.max_num_query = episode_descr_config.MAX_NUM_QUERY
    self.max_support_set_size = episode_descr_config.MAX_SUPPORT_SET_SIZE
    self.max_support_size_contrib_per_class = episode_descr_config.MAX_SUPPORT_SIZE_CONTRIB_PER_CLASS
    self.min_log_weight = episode_descr_config.MIN_LOG_WEIGHT
    self.max_log_weight = episode_descr_config.MAX_LOG_WEIGHT
    self.min_examples_in_class = episode_descr_config.MIN_EXAMPLES_IN_CLASS
    self.use_dag_hierarchy = episode_descr_config.USE_DAG_HIERARCHY
    self.use_bilevel_hierarchy = episode_descr_config.USE_BILEVEL_HIERARCHY

    # all class ids
    if dataset_spec["name"] == "Omniglot":
      self.class_set = get_bilevel_classes(self.split, self.dataset_spec)
    else:
      self.class_set = get_classes(self.split, self.dataset_spec["num_classes_per_split"])

    self.num_classes = len(self.class_set)
    # Filter out classes with too few examples
    self._filtered_class_set = []
    # Store (class_id, n_examples) of skipped classes for logging.
    skipped_classes = []
    
    for class_id in self.class_set:
      n_examples = len(dataset_spec["images_per_class"][class_id])

      if n_examples < self.min_examples_in_class:
        skipped_classes.append((class_id, n_examples))
      else:
        self._filtered_class_set.append(class_id)
    self.num_filtered_classes = len(self._filtered_class_set)

    if skipped_classes:
      logging.info(
          'Skipping the following classes, which do not have at least '
          '%d examples', self.min_examples_in_class)
    for class_id, n_examples in skipped_classes:
      logging.info('%s (ID=%d, %d examples)',
                   dataset_spec["id2name"][class_id], class_id, n_examples)

    if self.min_ways and self.num_filtered_classes < self.min_ways:
      raise ValueError(
          '"min_ways" is set to {}, but split {} of dataset {} only has {} '
          'classes with at least {} examples ({} total), so it is not possible '
          'to create an episode for it. This may have resulted from applying a '
          'restriction on this split of this dataset by specifying '
          'benchmark.restrict_classes or benchmark.min_examples_in_class.'
          .format(self.min_ways, split, dataset_spec["name"],
                  self.num_filtered_classes, self.min_examples_in_class,
                  self.num_classes))


    # for ImageNet
    if self.dataset_spec["name"] == "ILSVRC" and self.use_dag_hierarchy:
      if self.num_ways is not None:
        raise ValueError('"use_dag_hierarchy" is incompatible with "num_ways".')

      if not dataset_spec["name"] == "ILSVRC":
        raise ValueError('Only applicable to ImageNet.')

      # A DAG for navigating the ontology for the given split.
      graph = dataset_spec["split_subgraph"][self.split]

      # Map the absolute class IDs in the split's class set to IDs relative to
      # the split.
      class_set = self.class_set
      abs_to_rel_ids = dict((abs_id, i) for i, abs_id in enumerate(class_set))

      # Extract the sets of leaves and internal nodes in the DAG.
      leaves = set(get_leaves(graph))
      internal_nodes = graph - leaves  # set difference

      # Map each node of the DAG to the Synsets of the leaves it spans.
      spanning_leaves_dict = get_spanning_leaves(graph)

      # Build a list of lists storing the relative class IDs of the spanning
      # leaves for each eligible internal node. We ensure a deterministic order
      # by sorting the inner-nodes and their corresponding leaves by wn_id.
      self.span_leaves_rel = []
      for node in sorted(internal_nodes, key=lambda n: n.wn_id):
        node_leaves = sorted(spanning_leaves_dict[node], key=lambda n: n.wn_id)
        # Build a list of relative class IDs of leaves that have at least
        # min_examples_in_class examples.
        ids_rel = []
        for leaf in node_leaves:
          abs_id = dataset_spec["class_names_to_ids"][leaf.wn_id]
          if abs_id in self._filtered_class_set:
            ids_rel.append(abs_to_rel_ids[abs_id])

        # Internal nodes are eligible if they span at least
        # `min_allowed_classes` and at most `max_eligible` leaves.
        if self.min_ways <= len(ids_rel) <= MAX_SPANNING_LEAVES_ELIGIBLE:
          self.span_leaves_rel.append(ids_rel)

      num_eligible_nodes = len(self.span_leaves_rel)
      if num_eligible_nodes < 1:
        raise ValueError('There are no classes eligible for participating in '
                         'episodes. Consider changing the value of '
                         '`EpisodeDescriptionSampler.min_ways`, or '
                         'or MAX_SPANNING_LEAVES_ELIGIBLE')
    # For Omniglot.
    elif self.dataset_spec["name"] == "Omniglot" and self.use_bilevel_hierarchy:
      if self.num_ways is not None:
        raise ValueError('"use_bilevel_hierarchy" is incompatible with '
                         '"num_ways".')
      if self.min_examples_in_class > 0:
        raise ValueError('"use_bilevel_hierarchy" is incompatible with '
                         '"min_examples_in_class".')
      if not dataset_spec["name"] == "Omniglot":
        raise ValueError('"Only applicable to Omniglot."')

      # The id's of the superclasses of the split (a contiguous range of ints).
      all_superclasses = get_classes(self.split, self.dataset_spec["superclasses_per_split"])
      self.superclass_set = []
      for i in all_superclasses:
        if self.dataset_spec["classes_per_superclass"][i] < self.min_ways:
          raise ValueError(
              'Superclass: %d has num_classes=%d < min_ways=%d.' %
              (i, self.dataset_spec["classes_per_superclass"][i], self.min_ways))
        self.superclass_set.append(i)
                

  def sample_class_ids(self):
    """Returns the (relative) class IDs for an episode.
    If self.min_examples_in_class > 0, classes with too few examples will not
    be selected.
    """
    if self.dataset_spec["name"] == "ILSVRC" and self.use_dag_hierarchy:
      # Retrieve the list of relative class IDs for an internal node sampled
      # uniformly at random.
      index = self._rng.choice(list(range(len(self.span_leaves_rel))))
      episode_classes_rel = self.span_leaves_rel[index]

      # If the number of chosen classes is larger than desired, sub-sample them.
      if len(episode_classes_rel) > self.max_ways_upper_bound:
        episode_classes_rel = self._rng.choice(
            episode_classes_rel,
            size=[self.max_ways_upper_bound],
            replace=False)

      # Light check to make sure the chosen number of classes is valid.
      assert len(episode_classes_rel) >= self.min_ways
      assert len(episode_classes_rel) <= self.max_ways_upper_bound
    elif self.dataset_spec["name"] == "Omniglot" and self.use_bilevel_hierarchy:
      episode_superclass = self._rng.choice(self.superclass_set, 1)[0]
      num_superclass_classes = self.dataset_spec["classes_per_superclass"][
            episode_superclass]
      num_ways = sample_num_ways_uniformly(
          num_superclass_classes,
          min_ways=self.min_ways,
          max_ways=self.max_ways_upper_bound,
          rng=self._rng)

      # e.g. if these are [3, 1] then the 4'th and the 2'nd of the subclasses
      # that belong to the chosen superclass will be used. If the class id's
      # that belong to this superclass are [23, 24, 25, 26] then the returned
      # episode_classes_rel will be [26, 24] which as usual are number relative
      # to the split.

      episode_subclass_ids = sample_class_ids_uniformly(
          num_ways, num_superclass_classes, rng=self._rng)

      # The number of classes before the start of superclass_id, i.e. the class id
      # of the first (sub-)class of the given superclass.

      superclass_offset = sum([
        self.dataset_spec["classes_per_superclass"][superclass_id]
        for superclass_id in range(episode_superclass)
      ])

      # Absolute class ids (between 0 and the total number of dataset classes).
      class_ids = [superclass_offset + class_ind for class_ind in episode_subclass_ids]


      def _get_split_offset(split):
        """
        For Omniglot.
        Returns the starting class id of the contiguous chunk of ids of split.
        Args:
          split: A Split, the split for which to get classes.
        Raises:
          ValueError: Invalid dataset split.
        """
        if split == Split.TRAIN:
          offset = 0
        elif split == Split.VALID:
          previous_superclasses = range(
              0, self.dataset_spec["superclasses_per_split"][Split.TRAIN])
          offset = sum([
              self.dataset_spec["classes_per_superclass"][superclass_id]
              for superclass_id in previous_superclasses
          ])
        elif split == Split.TEST:
          previous_superclasses = range(
              0, self.dataset_spec["superclasses_per_split"][Split.TRAIN] +
              self.dataset_spec["superclasses_per_split"][Split.VALID])
          offset = sum([
              self.dataset_spec["classes_per_superclass"][superclass_id]
              for superclass_id in previous_superclasses
          ])
        else:
          raise ValueError('Invalid dataset split.')
        return offset
      # Relative (between 0 and the total number of classes in the split).
      # This makes the assumption that the class id's are in a contiguous range.
      episode_classes_rel = [
          class_id - _get_split_offset(self.split) for class_id in class_ids
      ]
    else:
      if self.num_ways is not None:
          num_ways = self.num_ways
      else:
          num_ways = sample_num_ways_uniformly(
              self.num_filtered_classes,
              min_ways=self.min_ways,
              max_ways=self.max_ways_upper_bound,
              rng=self._rng)
      # Filtered class IDs relative to the selected split
      ids_rel = [
          class_id - self.class_set[0] for class_id in self._filtered_class_set
      ]
      episode_classes_rel = sample_class_ids_uniformly(
          num_ways, ids_rel, rng=self._rng)

    return episode_classes_rel

  def sample_single_episode(self, sequential_sampling = False):


      class_ids = self.sample_class_ids()

      #cid: relative. self.class_set[cid]: absolute.
      num_images_per_class = np.array([
        len(self.dataset_spec["images_per_class"][self.class_set[cid]]) for cid in class_ids
        ])

      if self.num_query is not None:
        num_query = self.num_query
      else:
        num_query = compute_num_query(
            num_images_per_class,
            max_num_query=self.max_num_query,
            num_support=self.num_support)
      
      if self.num_support is not None:
        if isinstance(self.num_support, int):
            if any(self.num_support + num_query > num_images_per_class):
                raise ValueError('Some classes do not have enough examples.')
            num_support = self.num_support
        else:
            start, end = self.num_support
            if any(end + num_query > num_images_per_class):
                raise ValueError('The range provided for uniform sampling of the '
                            'number of support examples per class is not valid: '
                            'some classes do not have enough examples.')
            num_support = self._rng.randint(low=start, high=end + 1)
        num_support_per_class = np.array([num_support for _ in class_ids])
      else:
        num_remaining_per_class = num_images_per_class - num_query
        support_set_size = sample_support_set_size(
            num_remaining_per_class,
            self.max_support_size_contrib_per_class,
            max_support_set_size=self.max_support_set_size,
            rng=self._rng)

        num_support_per_class = sample_num_support_per_class(
            num_images_per_class,
            num_remaining_per_class,
            support_set_size,
            min_log_weight=self.min_log_weight,
            max_log_weight=self.max_log_weight,
            rng=self._rng)  

      total_num_per_class = num_query+num_support_per_class

      # class id in the task
      in_task_class_id = 0

      images = collections.defaultdict(list)
      labels = collections.defaultdict(list)

      for i, cid in enumerate(class_ids):

        # the sequential sampling of images of original Meta-Dataset. 
        if sequential_sampling:
          all_selected_files = self.dataset_spec["images_per_class"][self.class_set[cid]][:total_num_per_class[i]]

          self.dataset_spec["images_per_class"][self.class_set[cid]] = self.dataset_spec["images_per_class"][self.class_set[cid]][total_num_per_class[i]:]+all_selected_files
        else:
          # random sampling of images.
          all_selected_files = self._rng.choice(self.dataset_spec["images_per_class"][self.class_set[cid]],
                                                total_num_per_class[i], False)

        for file_ in all_selected_files[total_num_per_class[i]-num_query:]:
            images["query"].append(file_)
            labels["query"].append(torch.tensor([in_task_class_id]))
        
            
        
        for file_ in all_selected_files[:total_num_per_class[i]-num_query]:
            images["support"].append(file_)
            labels["support"].append(torch.tensor([in_task_class_id]))


        in_task_class_id += 1

      labels["query"] = torch.stack(labels["query"])
      labels["support"] = torch.stack(labels["support"])
      return images, labels             

  def sample_multiple_episode(self, batchsize, sequtial_sampling = False):
      all_images = []
      all_labels = []

      
      for task_index in range(batchsize):
        images, labels = self.sample_single_episode(sequtial_sampling)
        all_images.append(images)
        all_labels.append(labels)

      return all_images, all_labels


class BatchSampler(object):
  """Generates samples of a simple batch.
  In particular, for each batch, it will sample all files and labels of that batch.
  """
  def __init__(self, seed, dataset_spec, split):
    """
    seed: seed for sampling
    dataset_spec: dataset specification
    split: which split to sample from
    """

    # Fixing seed for sampling
    self._rng = np.random.RandomState(seed)



    self.dataset_spec = dataset_spec
    self.split = split

    # all class ids
    if dataset_spec["name"] == "Omniglot":
      self.class_set = get_bilevel_classes(self.split, self.dataset_spec)
    else:
      self.class_set = get_classes(self.split, self.dataset_spec["num_classes_per_split"])
    
    # all files
    self.all_file_path = []
    self.all_labels = []
    
    for class_id in self.class_set:
      self.all_file_path.extend(dataset_spec["images_per_class"][class_id])
      self.all_labels.extend([class_id]*len(dataset_spec["images_per_class"][class_id]))
    self.length = len(self.all_file_path)

    self.init()

  def init(self):
    self.batch_id = 0
    
  def shuffle_data(self):
    indexes = list(range(self.length))

    # random shuffle
    self._rng.shuffle(indexes)
    self.all_file_path = [i for _,i in sorted(zip(indexes,self.all_file_path))]
    self.all_labels = [i for _,i in sorted(zip(indexes,self.all_labels))]



  def sample_batch(self, batch_size, shuffle = True):
    # reset batch_id
    if self.batch_id*batch_size>=self.length:
      self.init()

    # Shuffle the data after completing a round of the dataset
    if shuffle and self.batch_id == 0:
      self.shuffle_data()


    file_paths = self.all_file_path[self.batch_id*batch_size:min(self.length, (self.batch_id+1)*batch_size)]
    labels = torch.tensor(self.all_labels[self.batch_id*batch_size:min(self.length, (self.batch_id+1)*batch_size)])

    images = []
    for file_ in file_paths:
      images.append(file_)

    self.batch_id += 1
    return images, labels


    






