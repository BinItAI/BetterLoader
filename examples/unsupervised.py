import sys
import os

sys.path.append(os.getcwd())
sys.path.append('..')

from betterloader import UnsupervisedBetterLoader
from betterloader.defaults import _simple, _collate
from PIL import Image



def produce_metadata():
    train_test_val_instances, classdata, pretransform = _simple()
    basic_collate_fn = _collate()

    metadata = {}
    metadata["pretransform"] = pretransform
    metadata["classdata"] = classdata
    metadata["train_test_val_instances"] = train_test_val_instances
    metadata["supervised"] = False
    metadata["custom_collate"] = basic_collate_fn
    metadata["drop_last"] = True
    metadata["eccentric_object"] = False
    metadata["sample_type"] = 'subset_sampling'

    return metadata


def run():


    index_json = "./sample_index_unsupervised.json"
    basepath = "./sample_dataset/"
    batch_size = 2
    metadata = produce_metadata()
    better_loader = UnsupervisedBetterLoader(basepath=basepath, base_experiment_details=['simclr', 1, (150, 150)],
                                             index_json_path=index_json, dataset_metadata=metadata)
    dataloaders, sizes = better_loader.fetch_segmented_dataloaders(batch_size=batch_size)

    for i, ((xp1, xp2), _) in enumerate(dataloaders['train']):
        print(i)
        print(xp1.shape)


if __name__ == '__main__':

    run()

