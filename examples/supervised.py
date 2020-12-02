"""
A simple betterloader.SupervisedLoader usage example. We should really add to these.
Notes:
1. This example uses the default parameters.
2. The structure of the index file can be any json as long as you change the other parameters to work with that
"""

from betterloader import SupervisedLoader

index_json = "./examples/sample_index.json"
basepath = "./examples/sample_dataset/"
batch_size = 2

loader = SupervisedLoader(basepath=basepath, index_json_path=index_json)
dataloaders, sizes = loader.fetch_segmented_dataloaders(
    batch_size=batch_size, transform=None
)

print("Dataloader sizes: {}".format(str(sizes)))
