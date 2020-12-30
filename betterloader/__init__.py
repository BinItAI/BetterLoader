"""
Adding Hypercustomizability to the Pytorch ImageFolder Dataloader.
id est: making it harder to do easy things, but easier to do harder things :)

"""

__version__ = "0.1.3"
__author__ = "BinIt Inc"
__credits__ = "N/A"

import json
import os
from collections import defaultdict
import torchvision.transforms as transforms
from .standard_transforms import GaussianBlur, TransformWhileSampling
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import torchvision

from .ImageFolderCustom import ImageFolderCustom
from .defaults import simple_metadata


def fetch_json_from_path(path):
    """Helper method to fetch json dict from file
    Args:
        path: Path to fetch json dict
    Returns:
        dict: JSON object stored in file at path
    """
    if path is not None:
        with open(path, "r") as file:
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


class BetterLoader:  # pylint: disable=too-few-public-methods
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
        dataset_metadata (dict, optional): Optional metadata parameters.  This dictionary must atleast
        contain a bool indicating if the experiment is 'supervised', and addtionally might contain 'custom_collate'
        (a custom collator), an additional 'sample_type' for arbitrary data sampling, 'eccentric_object' to indicate
        if it has to be pinned in memory and a indicator to 'drop_last' for non-integer  sample_size/batch_size value
        split (tuple): Tuple of train test val float values

    """

    def __init__(
        self,
        basepath,
        index_json_path=None,
        num_workers=1,
        index_object=None,
        subset_json_path=None,
        subset_object=None,
        dataset_metadata=None,
    ):
        if not os.path.exists(basepath):
            raise FileNotFoundError("Please supply a valid path to your base folder!")

        if index_json_path and not os.path.exists(index_json_path):
            if not index_object:
                raise FileNotFoundError(
                    "Please supply a valid path to a dataset index file or valid index object!"
                )
        if index_object and index_json_path:
            raise ValueError(
                "you must only define either the index_json_path or index object, not both!"
            )
        if subset_object and subset_json_path:
            raise ValueError(
                "you must only define either the subset_json_path or subset object, not both!"
            )
        self.basepath = basepath
        self.num_workers = num_workers
        self.subset_json_path = subset_json_path
        self.index_json_path = index_json_path
        self.subset_object = subset_object
        self.index_object = index_object
        self.classes = []
        self.class_to_idx = {}
        self.dataset_metadata = (
            {}
            if dataset_metadata is None
            else {i: dataset_metadata[i] for i in dataset_metadata if i != "split"}
        )
        self.split = (
            self.dataset_metadata["split"]
            if "split" in self.dataset_metadata
            else (0.6, 0.2, 0.2)
        )
        self.supervised = (
            self.dataset_metadata["supervised"]
            if "supervised" in self.dataset_metadata
            else True
        )
        self.custom_collator = (
            self.dataset_metadata["custom_collate"]
            if "custom_collate" in self.dataset_metadata
            else None
        )
        self.drop_last = (
            self.dataset_metadata["drop_last"]
            if "drop_last" in self.dataset_metadata
            else False
        )
        self.pin_mem = (
            self.dataset_metadata["eccentric_object"]
            if "eccentric_object" in self.dataset_metadata
            else False
        )
        self.sampler = (
            self.dataset_metadata["sample_type"]
            if "sample_type" in self.dataset_metadata
            else None
        )

        # self.dataloader_params = self.dataset_metadata['dataloader_params'] if 'dataloader_params' in self.dataset_metadata else {}

    def _set_class_data(self, datasets):
        """Wrapper to set class data values upon processing datasets
        Args:
                list: datasets that have been processed
        """

        if not (
            all(x.classes == datasets[0].classes for x in datasets)
            and all(x.class_to_idx == datasets[0].class_to_idx for x in datasets)
        ):
            print(
                "Class mismatch between the train/test/val data. This is usually caused by an uneven split, or a lack of the presence of identical classes in train/test/val. Assigning train data class names and class_to_idx map."
            )

        self.classes = datasets[0].classes
        self.class_to_idx = datasets[0].class_to_idx

    def _fetch_metadata(self, key):
        """Wrapper to fetch a value from the dataset metadata
        Args:
                key (string): Key that we're trying to fetch

        Returns:
                var: Value for that key - if such a key does not exist, return None
        """

        if key in self.dataset_metadata:
            return self.dataset_metadata[key]

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

        train_test_val_instances, class_data, pretransform = (
            self._fetch_metadata("train_test_val_instances"),
            self._fetch_metadata("classdata"),
            self._fetch_metadata("pretransform"),
        )



        if train_test_val_instances is None:
            train_test_val_instances = simple_metadata()["train_test_val_instances"]

        index, subset_json = (
            fetch_json_from_path(self.index_json_path)
            if self.index_json_path
            else self.index_object,
            fetch_json_from_path(self.subset_json_path)
            if self.subset_json_path
            else self.subset_object,
        )

        datasets = None

        def train_test_val_instances_wrap(
            directory, class_to_idx, index, is_valid_file
        ):
            return train_test_val_instances(
                self.split, directory, class_to_idx, index, is_valid_file
            )

        if transform is None:
            datasets = [
                ImageFolderCustom(
                    root=self.basepath,
                    is_valid_file=check_valid(subset_json),
                    instance=x,
                    index=index,
                    train_test_val_instances=train_test_val_instances_wrap,
                    class_data=class_data,
                    pretransform=pretransform,
                )
                for x in ("train", "test", "val")
            ]
        else:

            datasets = [
                ImageFolderCustom(
                    root=self.basepath,
                    transform=transform,
                    is_valid_file=check_valid(subset_json),
                    instance=x,
                    index=index,
                    train_test_val_instances=train_test_val_instances_wrap,
                    class_data=class_data,
                    pretransform=pretransform,
                )
                for x in ("train", "test", "val")
            ]

        self._set_class_data(datasets)

        custom_collator = self.custom_collator
        drop_last = self.drop_last
        pin_mem = self.pin_mem
        sampler = self.sampler

        if sampler is not None:
            dataloaders = [
                torch.utils.data.DataLoader(
                    x,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    collate_fn=custom_collator,
                    sampler=SubsetRandomSampler(list(range(len(x)))) if sampler == 'subset_sampling' else None,
                    pin_memory=pin_mem,
                    drop_last=drop_last,
                    shuffle=False
                )
                for x in [datasets[0]]
            ] + [
                torch.utils.data.DataLoader(
                    x,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    collate_fn=custom_collator,
                    sampler=SubsetRandomSampler(list(range(len(x)))) if sampler == 'subset_sampling' else None,
                    pin_memory=pin_mem,
                    drop_last=drop_last,
                )
                for x in datasets[1:]
            ]

        else:
            dataloaders = [
                torch.utils.data.DataLoader(
                    x,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    collate_fn=custom_collator,
                    shuffle=True,
                    pin_memory=pin_mem,
                    drop_last=drop_last,
                )
                for x in [datasets[0]]
            ] + [
                torch.utils.data.DataLoader(
                    x,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    collate_fn=custom_collator,
                    shuffle=False,
                    pin_memory=pin_mem,
                    drop_last=drop_last,
                )
                for x in datasets[1:]
            ]

        loaders = {
            "train": dataloaders[0],
            "test": dataloaders[1],
            "val": dataloaders[2],
        }

        sizes = {
            "train": len(datasets[0]),
            "test": len(datasets[1]),
            "val": len(datasets[2]),
        }

        return loaders, sizes

class UnsupervisedBetterLoader(BetterLoader):



    def __init__(self,
                 basepath,
                 base_experiment_details,
                 index_json_path=None,
                 num_workers=1,
                 index_object=None,
                 subset_json_path=None,
                 subset_object=None,
                 dataset_metadata=None,
                 ):

        super(UnsupervisedBetterLoader, self).__init__(
            basepath,
            index_json_path,
            num_workers,
            index_object,
            subset_json_path,
            subset_object,
            dataset_metadata,
        )

        print("On computer version")
        self.base_experiment_name = base_experiment_details[0] if base_experiment_details is not None else 'simclr'
        self.experiment_transform_params = base_experiment_details[1:]

        self.setup_sampling()
        self.transforms = self.setup_transform()



    def setup_sampling(self):

        if self.base_experiment_name == 'simclr':

            if self.dataset_metadata['sample_type'] is None:
                self.dataset_metadata['sample_type'] = 'subset_sampling'

        else:
            raise Exception("Iteration of experiment is not currently supported")


    def setup_transform(self):

        if self.base_experiment_name == 'simclr':

            if not len(self.experiment_transform_params) == 2:
                raise Exception("For SimClR, experiment details should be of the form [experiment_name, side_jitter, input_shape]")

            side_jitter = self.experiment_transform_params[0]
            input_shape = self.experiment_transform_params[1]

            color_jitter = transforms.ColorJitter(0.8 * side_jitter, 0.8 * side_jitter, 0.8 * side_jitter,
                                                  0.2 * side_jitter)

            all_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=input_shape[0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=[int(0.1 * input_shape[0]), int(0.1 * input_shape[0])]),
                transforms.ToTensor()
            ])

            return TransformWhileSampling(all_transforms)



        else:

            raise Exception("Iteration of experiment is not currently supported")

    def fetch_segmented_dataloaders(self, batch_size):

        dataloaders, sizes = super(UnsupervisedBetterLoader, self).fetch_segmented_dataloaders(batch_size, self.transforms)
        return dataloaders, sizes
