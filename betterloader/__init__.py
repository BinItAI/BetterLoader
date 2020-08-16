"""
BetterLoader

<strikethrough>Fixing the Python DataLoader :)</strikethrough>

Adding Hypercustomizability to the Pytorch ImageFolder Dataloader
id est: making it harder to do easy things, but easier to do harder things :)

"""

__version__ = "0.1.0"
__author__ = 'BinIt Inc'
__credits__ = 'N/A'

from .ImageFolderCustom import ImageFolderCustom
import json
import torch
import torchvision
import os
from collections import defaultdict


def fetch_json_from_path(path):
	if path != None:
		with open(path, 'r') as file:
			return json.load(file)
	else:
		return None


def check_valid(subset_json):
    if subset_json == None:
        return lambda path: True

    def curry(image_path):
        if image_path in subset_json:
            return True
        return False
    return curry


def read_index_default(split, directory, class_to_idx, index, is_valid_file):
	train, test, val = [], [], []
	i = 0
	for target_class in sorted(class_to_idx.keys()):
		i += 1
		class_index = class_to_idx[target_class]
		if not os.path.isdir(directory):
			continue
		instances = []
		for file in index[target_class]:
			if is_valid_file(file):
				path = os.path.join(directory, file)
				# for each item in the instances the first value must be a resolvable path to the image
				# more data can be added to this tuple, this tuple becomes the values argument in the pretransform
				item = (path, class_index)
				instances.append(item)

		trainp, testp, valp = split

		train += instances[:int(len(instances)*trainp)]
		test += instances[int(len(instances)*trainp):int(len(instances)*(1-valp))]
		val += instances[int(len(instances)*(1-valp)):]
	return train, test, val

class BetterLoader:
	"""A hypercustomisable Python dataloader

    Args:
    	basepath (string): Root directory path.
    	index_json_path (string): Path to index file
    	num_workers (int, optional): Number of workers
    	subset_json_path (string, optional): Path to subset json
    	dataset_metadata (dict, optional): Optional metadata parameters.
    		Metadata keys
			pretransform: (callable, optional) Define a custom pretransform before images are loaded into the dataloader and transformed
			classdata: (callable, optional) Define a custom mapping for a custom format sample file to read data from the DatasetFolder class.
			split: (dict, optional) Tuple for train, test, val values - must add to 1
			train_test_val_instances: (callable, optional) Custom function to read values from the index file

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
	def __init__(self, basepath, index_json_path, num_workers=1, subset_json_path=None, dataset_metadata={}):
		if not os.path.exists(basepath):
			raise Exception("Please supply a valid path to your base folder!")

		if not os.path.exists(index_json_path):
			raise Exception("Please supply a valid path to a dataset index file!")

		self.basepath = basepath
		self.num_workers = num_workers
		self.subset_json_path = subset_json_path
		self.index_json_path = index_json_path
		self.classes = []
		self.class_to_idx = {}
		self.dataset_metadata = None if dataset_metadata == None else {i:dataset_metadata[i] for i in dataset_metadata if i!='split'}
		self.split = dataset_metadata["split"] if "split" in dataset_metadata else (0.6, 0.2, 0.2)


	def _set_class_data(self, datasets):
		if not (all(x.classes == datasets[0].classes for x in datasets) and all(x.class_to_idx == datasets[0].class_to_idx for x in datasets)):
			print("Class mismatch between the train/test/val data. This is usually caused by an uneven split, or a lack of the presence of identical classes in train/test/val. Assigning train data class names and class_to_idx map.")

		self.classes = datasets[0].classes
		self.class_to_idx = datasets[0].class_to_idx

	def _fetch_metadata(self, key):
		if key in self.dataset_metadata:
			return self.dataset_metadata[key]
		else:
			return None

	def fetch_segmented_dataloaders(self, batch_size, transform=None):
		"""Fetch custom dataloaders, which may be used with any PyTorch model

	    Args:
	    	batch_size (string): Image batch size.
	    	transform (callable, optional): PyTorch transform object

	     Return:
	        loaders (dict): A dictionary of dataloaders for train test split
	        sizes (dict): A dictionary of dataset sizes for train test split
	    """

		train_test_val_instances, class_data, pretransform = self._fetch_metadata("train_test_val_instances"), self._fetch_metadata("classdata"), self._fetch_metadata("pretransform")

		if train_test_val_instances == None:
			train_test_val_instances = read_index_default

		index, subset_json = fetch_json_from_path(self.index_json_path), fetch_json_from_path(self.subset_json_path)

		datasets = None

		train_test_val_instances_wrap = lambda directory, class_to_idx, index, is_valid_file: train_test_val_instances(self.split, directory, class_to_idx, index, is_valid_file)

		if transform == None:
			datasets = [ImageFolderCustom(root=self.basepath, is_valid_file=check_valid(subset_json),
				instance=x, index=index,  train_test_val_instances=train_test_val_instances_wrap, class_data=class_data, pretransform = pretransform) for x in ('train', 'test', 'val')]
		else:
			datasets = [ImageFolderCustom(root=self.basepath, transform=transform, is_valid_file=check_valid(subset_json),
				instance=x, index=index,  train_test_val_instances=train_test_val_instances_wrap, class_data=class_data, pretransform = pretransform) for x in ('train', 'test', 'val')]

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