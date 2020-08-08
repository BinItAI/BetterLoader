from BetterLoader import fetch_segmented_dataloader
from dataset_metadata import metadata


index_json = 'idx.json'
dataloaders = fetch_segmented_dataloader(basepath, batch_size, transform, num_workers=8, subset_json_path=None, index_json_path=index_json, dataset_metadata = metadata())
