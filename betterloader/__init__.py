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

class BetterLoader:
	def __init__(self, basepath, index_json_path, num_workers=1, subset_json_path=None):
		if not os.path.exists(basepath):
			raise Exception("Please supply a valid path to your base folder!")

		if not os.path.exists(index_json_path):
			raise Exception("Please supply a valid path to a dataset index file!")

		self.basepath = basepath
		self.num_workers = num_workers
		self.subset_json_path = subset_json_path
		self.index_json_path = index_json_path

	def fetch_segmented_dataloaders(self, batch_size, transform=None, dataset_metadata = None):
		'''Return a 2 element tuple, containing a list of dataloaders (train, test, split) along with a tuple containing their sizes'''
		
		train_test_val_instances, class_data, pretransform = dataset_metadata

		index, subset_json = fetch_json_from_path(self.index_json_path), fetch_json_from_path(self.subset_json_path)

		datasets = None

		if transform == None:
			datasets = [ImageFolderCustom(root=self.basepath, is_valid_file=check_valid(subset_json),
				instance=x, index=index,  train_test_val_instances=train_test_val_instances, class_data=class_data, pretransform = pretransform) for x in ('train', 'test', 'val')]
		else:
			datasets = [ImageFolderCustom(root=self.basepath, transform=transform, is_valid_file=check_valid(subset_json),
				instance=x, index=index,  train_test_val_instances=train_test_val_instances, class_data=class_data, pretransform = pretransform) for x in ('train', 'test', 'val')]
		
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