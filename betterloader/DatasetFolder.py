"""
Modified version of the PyTorch DatasetFolder class to make custom dataloading possible

"""

import os
import os.path

from torchvision.datasets import VisionDataset

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def default_pretransform(sample, values):
    """Returns the image sample without transforming it at all
    Args:
        sample: Loaded image data
        values: Tuple such that the 1th arguement is the target (defined by default)

    Returns:
        var: The loaded sample image
        int: Value representing the image class (label for data)
    """
    target = values[1]
    return sample, target

def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None, instance='train', index=None, train_test_val_instances=None):
    """Makes the actual dataset
    Args:
        directory (string): Root directory path.
        class_to_idx (dict): Dict which maps classes to index values
        extensions (tuple[string]): A list of allowed extensions. both extensions and is_valid_file should not be passed.
        is_valid_file (callable, optional): A function that takes path of a file and check if the file is a valid file (used to check of corrupt files) both extensions and is_valid_file should not be passed.
        instance (str): String signifying data segment (train, test, val)
        index (dict): Index file dict data
        train_test_val_instances (callable, optional): Returns custom breakup for train, test, val data

    Returns:
        list: List of PyTorch instance data to be loaded
    """
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    train, test, val = train_test_val_instances(directory, class_to_idx, index, is_valid_file)

    return train if instance == 'train' else test if instance == 'test' else val

class DatasetFolder(VisionDataset):
    """A generic data loader ::
    
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        instance (sting, optional): Either 'train' 'test' or 'val' whether or not you want the train test or val split
        index (dict[string:list[string]], optional): A dictionary that maps each class to a list of the image paths for that class along with whetever other data you need to make your dataset
            this can really be whatever you want because it is only handled by train_test_val_instances.
        train_test_val_instances (callable, optional): A function that takes:
            a root directory,
            a mapping of class names to indeces, 
            the index,
            and is_valid_file
            and returns a tuple of lists containing the instance data for each of train test and val, 
            the instance data in the list is a tuple and can have whatever structure you want as long as the image path is the first element
                each of these tuples is processed by the pretransform
        class_data (tuple, optional): the first element is a list of the classes, the second is a mapping of the classes to their indeces
        pretransform (callable, optional): A function that takes the loaded image and any other relevant data for that image and returns a transformed version of that image

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        pretransform (callable): returns a transformed image using data in the sample
        class_data (tuple): (classes, class_to_idx)
        samples (tuple): tuple of three (train test val) lists of (sample path, class_index, whatever else, ...) tuples
        Unused: targets (list): The class_index value for each image in the dataset 
    """

    def __init__(self, root, loader, extensions, transform,
                 target_transform, is_valid_file, instance,
                 index, train_test_val_instances, class_data, pretransform):

        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.index = index
        self.class_data = class_data
        self.pretransform = default_pretransform if pretransform is None else pretransform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file, instance, index, train_test_val_instances)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, root_dir):
        """
        Finds the class folders in a dataset.

        Args:
            root_dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        classes, class_to_idx = self.class_data(root_dir, self.index)

        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path, target, pt = self.samples[index]


        values = self.samples[index]
        path = values[0]
        sample = self.loader(path)
        sample, target = self.pretransform(sample, values)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)
