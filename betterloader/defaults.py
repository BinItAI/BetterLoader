"""A collection of aggregated default methods and metadata accessors, used for both testing and default values
"""

import os
import re
import torch
from torch._six import container_abcs, string_classes, int_classes


def _simple():
    def train_test_val_instances(
        split, directory, class_to_idx, index, is_valid_file
    ):  # pylint: disable=too-many-locals
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

            train += instances[: int(len(instances) * trainp)]
            test += instances[
                int(len(instances) * trainp) : int(len(instances) * (1 - valp))
            ]
            val += instances[int(len(instances) * (1 - valp)) :]
        return train, test, val

    def classdata(_, index):
        """Given class data, just create the default classes list and class_to_idx dict"""
        classes = list(index.keys())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def pretransform(sample, values):
        """Given a sample and a values list as specified in the docs, just return the path"""
        target = values[1]
        return sample, target

    return train_test_val_instances, classdata, pretransform


def _regex():
    def train_test_val_instances(
        split, directory, class_to_idx, index, is_valid_file
    ):  # pylint: disable=too-many-locals
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

        def _fetch_regex_names(regex):
            files = []
            for filename in os.listdir(directory):
                if re.compile(regex).match(filename):
                    files.append(filename)
            return files

        for target_class in sorted(class_to_idx.keys()):
            i += 1
            regex = index[target_class]
            if not os.path.isdir(directory):
                continue
            instances = []
            files = _fetch_regex_names(regex)
            for file in files:
                if is_valid_file(file):
                    path = os.path.join(directory, file)
                    instances.append((path, class_to_idx[target_class]))

            trainp, _, valp = split

            train += instances[: int(len(instances) * trainp)]
            test += instances[
                int(len(instances) * trainp) : int(len(instances) * (1 - valp))
            ]
            val += instances[int(len(instances) * (1 - valp)) :]
        return train, test, val

    def classdata(_, index):
        """Given class data, just create the default classes list and class_to_idx dict"""
        classes = list(index.keys())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def pretransform(sample, values):
        """Given a sample and a values list as specified in the docs, just return the path"""
        target = values[1]
        return sample, target

    return train_test_val_instances, classdata, pretransform


def _collate():
    np_str_obj_array_pattern = re.compile(r"[SaUO]")
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}"
    )

    def basic_collate_fn(batch):
        """Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return basic_collate_fn([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: basic_collate_fn([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(*(basic_collate_fn(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    "each element in list of batch should be of equal size"
                )
            transposed = zip(*batch)
            return [basic_collate_fn(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))

    return basic_collate_fn


def simple_metadata():
    """Create a very simple metadata object to test with"""
    train_test_val_instances, classdata, pretransform = _simple()
    metadata = {}
    metadata["pretransform"] = pretransform
    metadata["classdata"] = classdata
    metadata["train_test_val_instances"] = train_test_val_instances
    return metadata


def regex_metadata():
    """Create a regex based metadata object"""
    train_test_val_instances, classdata, pretransform = _regex()
    metadata = {}
    metadata["pretransform"] = pretransform
    metadata["classdata"] = classdata
    metadata["train_test_val_instances"] = train_test_val_instances
    return metadata


def collate_metadata():
    """Create a collation based metadata object"""
    train_test_val_instances, classdata, pretransform = _simple()
    basic_collate_fn = _collate()
    metadata = {}
    metadata["pretransform"] = pretransform
    metadata["classdata"] = classdata
    metadata["train_test_val_instances"] = train_test_val_instances
    metadata["supervised"] = True
    metadata["custom_collate"] = basic_collate_fn
    metadata["drop_last"] = True
    metadata["eccentric_object"] = False
    metadata["sample_type"] = None

    return metadata
