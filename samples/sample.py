from BetterLoader import fetch_segmented_dataloader

#index needs to be formatted as a dictionary where the keys are the classes and the values are lists where each list item is the required information to use each image
#the image path is the only neccessary value, but other information can be used
index_json = 'sample_index.json'

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
  
def metadata():
  return traintestval, classdata, pretransform
  
    
basepath = "something/something_else/"
batch_size = 2
transform = '' #some pytorch transform
dataloaders = fetch_segmented_dataloader(basepath, batch_size, transform, num_workers=8, subset_json_path=None, index_json_path=index_json, dataset_metadata = metadata())
