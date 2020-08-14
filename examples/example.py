from betterloader import BetterLoader
import os

index_json = './examples/sample_index.json'
basepath = "./examples/sample_dataset/"
batch_size = 2

def classdata(dir, index):
    classes = list(index.keys())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
 
def pretransform(sample, values):
    # since the tuple we defined in traintestval has the target in the 1 index
    target = values[1]
    return sample,target

'''
Metadata keys:
classdata: @James can you write a short summary for this
pretransform: @James can you write a short summary for this
split: Optional tuple for train, test, val values - must add to 1
train_test_val_instances: Optional custom function to read values from the index file
'''
dataset_metadata = { "classdata": classdata, "pretransform": pretransform }
better_loader = BetterLoader(basepath=basepath, index_json_path=index_json, num_workers=1, subset_json_path=None, dataset_metadata=dataset_metadata)
dataloaders, sizes = better_loader.fetch_segmented_dataloaders(batch_size=batch_size, transform=None)

print("Dataloader sizes: {}".format(str(sizes)))