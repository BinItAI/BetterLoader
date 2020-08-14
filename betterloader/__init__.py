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

def _read_index(split, directory, class_to_idx, index, is_valid_file):
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
	def __init__(self, basepath, index_json_path, num_workers=1, subset_json_path=None, dataset_metadata=None):
		if not os.path.exists(basepath):
			raise Exception("Please supply a valid path to your base folder!")

		if not os.path.exists(index_json_path):
			raise Exception("Please supply a valid path to a dataset index file!")

		self.basepath = basepath
		self.num_workers = num_workers
		self.subset_json_path = subset_json_path
		self.index_json_path = index_json_path
		self.dataset_metadata = None if dataset_metadata == None else {i:dataset_metadata[i] for i in dataset_metadata if i!='split'}
		self.split = dataset_metadata["split"] if "split" in dataset_metadata else (0.6, 0.2, 0.2)

	def _fetch_metadata(self, key):
		if key in self.dataset_metadata:
			return self.dataset_metadata[key]
		else:
			return None

	def fetch_segmented_dataloaders(self, batch_size, transform=None):
		'''Return a 2 element tuple, containing a list of dataloaders (train, test, split) along with a tuple containing their sizes'''

		train_test_val_instances, class_data, pretransform = self._fetch_metadata("train_test_val_instances"), self._fetch_metadata("classdata"), self._fetch_metadata("pretransform")

		if train_test_val_instances == None:
			train_test_val_instances = _read_index

		index, subset_json = fetch_json_from_path(self.index_json_path), fetch_json_from_path(self.subset_json_path)

		datasets = None

		train_test_val_instances_wrap = lambda directory, class_to_idx, index, is_valid_file: train_test_val_instances(self.split, directory, class_to_idx, index, is_valid_file)

		if transform == None:
			datasets = [ImageFolderCustom(root=self.basepath, is_valid_file=check_valid(subset_json),
				instance=x, index=index,  train_test_val_instances=train_test_val_instances_wrap, class_data=class_data, pretransform = pretransform) for x in ('train', 'test', 'val')]
		else:
			datasets = [ImageFolderCustom(root=self.basepath, transform=transform, is_valid_file=check_valid(subset_json),
				instance=x, index=index,  train_test_val_instances=train_test_val_instances_wrap, class_data=class_data, pretransform = pretransform) for x in ('train', 'test', 'val')]

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