"""
Adding Hypercustomizability to the Pytorch ImageFolder Dataloader.
id est: making it harder to do easy things, but easier to do harder things :)

"""

__version__ = "0.1.3"
__author__ = 'BinIt Inc'
__credits__ = 'N/A'

import json
import os
from collections import defaultdict

import torch
import torchvision

from .ImageFolderCustom import ImageFolderCustom

def fetch_json_from_path(path):
    """Helper method to fetch json dict from file
    Args:
        path: Path to fetch json dict
    Returns:
        dict: JSON object stored in file at path
    """
    if path is not None:
        with open(path, 'r') as file:
            return json.load(file)
    else:
        return None


def check_valid(subset_json):
    """Helper to check if a file is valid, given a subset json instance
    Args:
        subset_json: Defined subset json file data
    Returns:
        bool: True/false value for validity
    """
    if subset_json is None:
        return lambda path: True

    def curry(image_path):
        if image_path in subset_json:
            return True
        return False
    return curry


def read_index_default(split, directory, class_to_idx, index, is_valid_file): # pylint: disable=too-many-locals
    """Function to perform default train/test/val instance creation
    Args:
        split (tuple): Tuple of ratios (from 0 to 1) for train, test, val values
        directory (str): Parent directory to read images from
        class_to_idx (dict): Dictionary to map values from class strings to index values
        index (dict): Index file dict object
        is_valid_file (callable): Function to verify if a file should be loaded
    Returns:
        (tuple): Tuple of length 3 containing train, test, val instances
    """
    train, test, val = [], [], []
    i = 0
    for target_class in sorted(class_to_idx.keys()):
        i += 1
        if not os.path.isdir(directory):
            continue
        instances = []
        for file in index[target_class]:
            if is_valid_file(file):
                path = os.path.join(directory, file)
                instances.append((path, class_to_idx[target_class]))

        trainp, _, valp = split

        train += instances[:int(len(instances)*trainp)]
        test += instances[int(len(instances)*trainp):int(len(instances)*(1-valp))]
        val += instances[int(len(instances)*(1-valp)):]
    return train, test, val


class BetterLoader: # pylint: disable=too-few-public-methods
    """A hypercustomisable Python dataloader

    Args:
        basepath (string): Root directory path.
        index_json_path (string): Path to index file
        num_workers (int, optional): Number of workers
        subset_json_path (string, optional): Path to subset json
        dataset_metadata (dict, optional): Optional metadata parameters:

            - pretransform (callable, optional): This allows us to load a custom pretransform before images are loaded into the dataloader and transformed.
            - classdata (callable, optional): Defines a custom mapping for a custom format index file to read data from the DatasetFolder class
            - split (tuple, optional): Defines a tuple for train, test, val values which must add to one
            - train_test_val_instances (callable, optional): Defines a custom function to read values from the index file

     Attributes:
        basepath (string): Root directory path.
        num_workers (int, optional): Number of workers
        subset_json_path (string, optional): Path to subset json
        index_json_path (string): Path to index file
        classes (list): List of the class names sorted alphabetically
        class_to_idx (dict): Dict with items (class_name, class_index).
        dataset_metadata (dict, optional): Optional metadata parameters. Also contains metadata key which has dataloader_params involved, which is a dict of additional required params.
        split (tuple): Tuple of train test val float values
    """

    def __init__(self, basepath, index_json_path, num_workers=1, subset_json_path=None, dataset_metadata=None):
        if not os.path.exists(basepath):
            raise Exception("Please supply a valid path to your base folder!")

        if not os.path.exists(index_json_path):
            raise Exception(
                "Please supply a valid path to a dataset index file!")

        self.basepath = basepath
        self.num_workers = num_workers
        self.subset_json_path = subset_json_path
        self.index_json_path = index_json_path
        self.classes = []
        self.class_to_idx = {}
        self.dataset_metadata = {} if dataset_metadata is None else {i: dataset_metadata[i] for i in dataset_metadata if i != 'split'}
        self.split = self.dataset_metadata["split"] if "split" in self.dataset_metadata else (0.6, 0.2, 0.2)
        self.dataloader_params = self.dataset_metadata['dataloader_params'] if 'dataloader_params' in self.dataset_metadata else None
        self.is_classed = self._fetch_dataloader_param('supervised')



    def _set_class_data(self, datasets):
        '''Wrapper to set class data values upon processing datasets
        Args:
                list: datasets that have been processed
        '''

        if not (all(x.classes == datasets[0].classes for x in datasets) and all(x.class_to_idx == datasets[0].class_to_idx for x in datasets)):
            print("Class mismatch between the train/test/val data. This is usually caused by an uneven split, or a lack of the presence of identical classes in train/test/val. Assigning train data class names and class_to_idx map.")

        self.classes = datasets[0].classes
        self.class_to_idx = datasets[0].class_to_idx

    def _fetch_metadata(self, key):
        '''Wrapper to fetch a value from the dataset metadata
        Args:
                key (string): Key that we're trying to fetch

        Returns:
                var: Value for that key - if such a key does not exist, return None
        '''

        if key in self.dataset_metadata:
            return self.dataset_metadata[key]

        return None

    def _fetch_dataloader_param(self, key):

        if key in self.dataloader_params:
            return self.dataloader_params[key]

        return None

    def fetch_segmented_dataloaders(self, batch_size, transform=None):
        """Fetch custom dataloaders, which may be used with any PyTorch model

        Args:
            batch_size (string): Image batch size.
            transform (callable, optional): PyTorch transform object

         Returns:
            dict: A dictionary of dataloaders for train test split
            dict: A dictionary of dataset sizes for train test split
        """

        train_test_val_instances, class_data, pretransform = self._fetch_metadata(
            "train_test_val_instances"), self._fetch_metadata("classdata"), self._fetch_metadata("pretransform")

        if train_test_val_instances is None:
            train_test_val_instances = read_index_default

        index, subset_json = fetch_json_from_path(
            self.index_json_path), fetch_json_from_path(self.subset_json_path)

        datasets = None

        def train_test_val_instances_wrap(directory, class_to_idx, index, is_valid_file):
            return train_test_val_instances(self.split, directory, class_to_idx, index, is_valid_file)

        if transform is None:
            datasets = [ImageFolderCustom(root=self.basepath, is_valid_file=check_valid(subset_json),
                                          instance=x, index=index, train_test_val_instances=train_test_val_instances_wrap, class_data=class_data, pretransform=pretransform) for x in ('train', 'test', 'val')]
        else:
            datasets = [ImageFolderCustom(root=self.basepath, transform=transform, is_valid_file=check_valid(subset_json),
                                          instance=x, index=index, train_test_val_instances=train_test_val_instances_wrap, class_data=class_data, pretransform=pretransform) for x in ('train', 'test', 'val')]

        self._set_class_data(datasets)

        dataloaders = [torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=self.num_workers, shuffle=True) for x in [
            datasets[0]]] + [torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=self.num_workers, shuffle=False) for x in datasets[1:]]

        loaders = {
            "train": dataloaders[0],
            "test": dataloaders[1],
            "val": dataloaders[2]
        }

        sizes = {
            "train": len(datasets[0]),
            "test": len(datasets[1]),
            "val": len(datasets[2])
        }

        return loaders, sizes
