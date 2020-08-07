"""
BetterLoader

<strikethrough>Fixing the Python DataLoader :)</strikethrough>

Adding Hypercustomizability to the Pytorch ImageFolder Dataloader
id est: making it harder to do easy things, but easier to do harder things

"""

__version__ = "0.1.0"
__author__ = 'BinIt Inc'
__credits__ = 'N/A'

from ImageFolderCustom import ImageFolderCustom
import json
import torch
import torchvision

def fetch_json_from_path(path):
if path != None:
	with open(path, 'r') as file:
		return json.load(file)
else:
	return None

def check_valid(subset_json):
    if subset_json == None:
        return lambda path: True
        #print('No subset json specified, using entire dataset...')

    def curry(image_path):
        if image_path in subset_json:
            return True
        return False
    return curry

class BetterLoader:
	def __init__(self):
		pass

	def fetch_segmented_dataloaders(self, basepath, batch_size, transform, num_workers=8, subset_json_path=None, index_json_path=None, dataset_metadata = None):
		'''Return a 2 element tuple, containing a list of dataloaders (train, test, split) along with a tuple containing their sizes'''
		
		#when we use this we can just have a separate file that defines this metadata per dataset, as it should always be constant
		#we might want to add the index_json to the metadata
		 train_test_val_instances, class_data, pretransform = dataset_metadata


		#this code feels unnecessary rn, we should just enforce that the user supplies the index.
		#or we can put the index in the metadata
		'''if index_json_path == None:
        	print("No index path specified, using default index path...")
			_ensure_folder(DEFAULT_INDEX_PATH)
			cached_file = os.path.join(DEFAULT_INDEX_PATH, DEFAULT_INDEX)
			if os.path.exists(cached_file) and os.path.isfile(cached_file):
				index_json_path = cached_file
			else:
				_download_default_index_file(cached_file)
				index_json_path = cached_file
		else:
			print("Specified index json path, using the same...")
		'''

		subset_json = fetch_json_from_path(subset_json_path)

		#i have absolutely no idea why this code exists
		#subset_json = None if subset_json == None else set(subset_json)

		index = fetch_json_from_path(index_json_path)

		datasets = [ImageFolderCustom(root=basepath, transform=transform, is_valid_file=check_valid(
			subset_json), instance=x, index=index,  train_test_val_instances=train_test_val_instances, class_data=class_data, pretransform = pretransform) for x in ('train', 'test', 'val')]
		dataloaders = [torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=True) for x in [
			datasets[0]]] + [torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=num_workers, shuffle=False) for x in datasets[1:]]
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
		print(str(sizes))
		return loaders, sizes
			return None, None