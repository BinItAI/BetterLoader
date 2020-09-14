# This example uses the default parameters
# The structure of the index file can be any json as long as you change the other parameters to work with that

from betterloader import BetterLoader
import os

index_json = './examples/sample_index.json'
basepath = "./examples/sample_dataset/"
batch_size = 2

better_loader = BetterLoader(basepath=basepath, index_json_path=index_json)
dataloaders, sizes = better_loader.fetch_segmented_dataloaders(batch_size=batch_size, transform=None)

print("Dataloader sizes: {}".format(str(sizes)))
