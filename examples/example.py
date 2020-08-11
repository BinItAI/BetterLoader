from betterloader import BetterLoader
import os

index_json = './examples/sample_index.json'
basepath = "./examples/sample_dataset/"
batch_size = 2

def traintestval(directory, class_to_idx, index, is_valid_file):
    train, test, val = [], [], []
    i = 0
    for target_class in sorted(class_to_idx.keys()):
        i+=1
        class_index = class_to_idx[target_class]
        if not os.path.isdir(directory):
            continue
        instances = []
        for file in index[target_class]:
            if is_valid_file(file):
                path = os.path.join(directory,file)
                # for each item in the instances the first value must be a resolvable path to the image
                # more data can be added to this tuple, this tuple becomes the values argument in the pretransform
                item = (path, class_index)
                instances.append(item)

        trainp = 0.6
        testp = 0.2
        valp = 0.2

        train += instances[:int(len(instances)*trainp)]
        test += instances[int(len(instances)*trainp):int(len(instances)*(1-valp))]
        val += instances[int(len(instances)*(1-valp)):]
    return train,test,val

def classdata(dir, index):
    classes = list(index.keys())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
 
def pretransform(sample, values):
    # since the tuple we defined in traintestval has the target in the 1 index
    target = values[1]
    return sample,target

dataset_metadata = (traintestval, classdata, pretransform)
better_loader = BetterLoader(basepath=basepath, index_json_path=index_json, num_workers=1, subset_json_path=None)
dataloaders, sizes = better_loader.fetch_segmented_dataloaders(batch_size=batch_size, transform=None, dataset_metadata=dataset_metadata)

print("Dataloader sizes: {}".format(str(sizes)))
