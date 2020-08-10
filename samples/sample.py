from BetterLoader import fetch_segmented_dataloader
from dataset_metadata import metadata


index_json = 'idx.json'

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
    target = values[1]
    return sample,target
  
def metadata():
  return traintestval, classdata, pretransform
  
dataloaders = fetch_segmented_dataloader(basepath, batch_size, transform, num_workers=8, subset_json_path=None, index_json_path=index_json, dataset_metadata = metadata())
