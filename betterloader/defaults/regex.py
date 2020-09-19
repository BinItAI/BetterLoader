'''
Simple function definitions for very basic BetterLoader metadata
'''
import os
import re

def train_test_val_instances(split, directory, class_to_idx, index, is_valid_file): # pylint: disable=too-many-locals
    """Function to perform default train/test/val instance creation
    Args:
        split (tuple): Tuple of ratios (from 0 to 1) for train, test, val values
        directory (str): Parent directory to read images from
        class_to_idx (dict): Dictionary to map values from class strings to index values
        index (dict): Index file dict object
        is_valid_file (callable): Function to verify if a file should be loaded
    Returns:
        (tuple): Tuple of length 3 containing train, test, val instances
    """
    train, test, val = [], [], []
    i = 0

    def _fetch_regex_names(regex):
        files = []
        for filename in os.listdir(directory):
            if re.compile(regex).match(filename):
                files.append(filename)
        return files

    for target_class in sorted(class_to_idx.keys()):
        i += 1
        regex = index[target_class]
        if not os.path.isdir(directory):
            continue
        instances = []
        files = _fetch_regex_names(regex)
        for file in files:
            if is_valid_file(file):
                path = os.path.join(directory, file)
                instances.append((path, class_to_idx[target_class]))

        trainp, _, valp = split

        train += instances[:int(len(instances)*trainp)]
        test += instances[int(len(instances)*trainp):int(len(instances)*(1-valp))]
        val += instances[int(len(instances)*(1-valp)):]
    return train, test, val

def classdata(_, index):
    '''Given class data, just create the default classes list and class_to_idx dict
    '''
    classes = list(index.keys())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
 
def pretransform(sample, values):
    '''Given a sample and a values list as specified in the docs, just return the path
    '''
    target = values[1]
    return sample, target
