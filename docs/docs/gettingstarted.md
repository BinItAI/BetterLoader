---
id: gettingstarted
title: Getting Started
sidebar_label: Getting Started
slug: /
---

## Installation

### Python
The BetterLoader library is hosted on [pypi](https://pypi.org/) and can be installed via [pip](https://pip.pypa.io/en/stable/).
```bash
pip install betterloader
```

### From Source
For developers, BetterLoader's source may also be found at our [Github repository](https://github.com/BinItAI/BetterLoader). You can also install BetterLoader from source, but if you're just trying to use the package, pip is probably a far better bet.

## Why BetterLoader?
BetterLoader really shines when you're working with a dataset, and you want to load subsets of image classes conditionally. Say you have 3 folders of images, and you only want to load those images that conform to a specific condition, <b>or</b> those that are present in a pre-defined subset file. What if you want to load a specific set of crops per source image, given a set of source images? BetterLoader can do all this, and more.<br />
<b>Note:</b> BetterLoader currently only supports supervised deep learning tasks. Unsupervised learning support coming soon!

### Creating a BetterLoader
Using BetterLoader with its default parameters lets it function just like the regular Python dataloader. A few points worth noting are that:
- BetterLoader does not expect a nested folder structure. In its current iteration, files are expected to all be present in the root directory. This lets us use index files to define classes and labels dynamically, and vary them from experiment to experiment.
- <b>Every</b> instance of BetterLoader requires an index file to function. The default index file format maps class names to a list of image paths, but the index file can be any json file as long as you modify train_test_val_instances to parse it correctly; for example you could instead map class names to regex for the file paths and pass a train_test_val_instances that reads the files based on that regex. Sample index files may be found <a href="/docs/files">here</a>.

A sample use-case for BetterLoader may be found below. It's worth noting that at this point in time, the BetterLoader class has only one callable function.
```python
from betterloader import BetterLoader

index_json = './examples/sample_index.json'
# or index_object = {"class1":["image0.jpg","image1.jpg","image2.jpg","image3.jpg"],"class2":["image4.jpg","image5.jpg","image6.jpg","image7.jpg"]}
basepath = "./examples/sample_dataset/"
batch_size = 2

loader = BetterLoader(basepath=basepath, index_json_path=index_json)
# or loader = BetterLoader(basepath=basepath, index_object=index_object)
dataloaders, sizes = loader.fetch_segmented_dataloaders(batch_size=batch_size, transform=None)

print("Dataloader sizes: {}".format(str(sizes)))
```

### Constructor Parameters
| field        |      type      |   description | optional (datatype) |
| ------------- | :-----------: | -----: | -----------: |
| basepath      | str | path to image directory | no |
| index_json_path      | str | path to index file | yes (None) |
| index_object | dict| An object representation of an index file | yes (None) |
| num_workers      | int | number of workers | yes (1) |
| subset_json_path      | str | path to subset json file | yes (None) |
| subset_object | dict| An object representation of the subset file | yes (None) |
| dataset_metadata      |   metadata object for dataset    |   list of optional metadata attributes to customise the BetterLoader | yes (None) |

### Usage
The BetterLoader class' `fetch_segmented_dataloaders` function allows for a user to obtain a tuple of dictionaries, which are most commonly referenced as `(dataloaders, sizes)`. Each dictionary consequently contains `train`, `test`, and `val` keys, allowing for easy access to the dataloaders, as well as their sizes. The function header for the same may be found below:

```
def fetch_segmented_dataloaders(self, batch_size, transform=None)
"""Fetch custom dataloaders, which may be used with any PyTorch model
    Args:
    batch_size (string): Image batch size.
    transform (callable or dict, optional): PyTorch transform object. This parameter may also be a dict with keys of 'train', 'test', and 'val', in order to enable separate transforms for each split.
    Returns:
        dict: A dictionary of dataloaders for train test split
        dict: A dictionary of dataset sizes for train test split
"""
```

#### Dataset Metadata
BetterLoader accepts certain key value pairs as dataset metadata, in order to enable some custom functionality.
1. pretransform (callable, optional): This allows us to load a custom pretransform before images are loaded into the dataloader and transformed.
  For basic usage a pretransform that does not do anything (the default) is usually sufficient. An example use case for the customizability is listed below.
2. classdata (callable, optional): Defines a custom mapping for a custom format index file to read data from the DatasetFolder class.
  Since the index file may have any structure we need to ensure that the classes and a mapping from the classes to the index are always available.
  Returns a tuple (list of classes, dictionary mapping of class to index)
3. split (tuple, optional): Defines a tuple for train, test, val values which must add to one.
4. train_test_val_instances (callable, optional): Defines a custom function to read values from the index file.
  The default expects an index that is a dict mapping classes to a list of file paths, will need to be written custom for different index formats.
  Always must return train test and val splits, which each need to be a list of tuples, each tuple corresponding to one datapoint.
  The first element of this tuple must also be the filepath of the image for that datapoint.
  The default also has the target class index as the second element of this tuple, this is probably good for most use cases.
  Each of these datapoint tuples is passed as the `values` argument in the pretransform, any additional data necessary for transforming the datapoint before it is loaded can go in the datapoint tuple.
5. supervised (bool, optional): Defines whether or not the experiment is supervised
6. custom_collator (callable, optional): Custom function that merges a list of samples to form a mini-batch of Tensors
7. drop_last (bool, optional): Defines whether to drop the last incomplete batch if the dataset is not divisible by batch size to avoid sizing errors
8. pin_mem (bool, optional): Sets the data load to copy tensors into CUDA pinned memory before returning them, providing your data elements are not custom type
9. sampler (torch.utils.data.Sampler or `iterable`, optional): Can be used to define a custom strategy to draw data from the dataset

---

Here is an example of a `pretransform` and a `train_test_val_instances` designed to allow for a specified crop to be taken of each image.
<b>Notes</b>:

- The internals of the loader dictate that the elements of the `instances` variables generated from train_test_val_instances will become the `values` argument for a pretransform call, and the `sample` argument for pretransform is the image data loaded directly from the filepath in `values[0]` (or `instances[i][0]`).
- Since the index file here has a similar structure to the default we can get away with using the default classdata function, but index files that don't have the classes as keys of a dictionary will need a custom way of determining the classes.

```python
def pretransform(sample, values):
    """Example pretransform - takes an image and crops it based on the parameters defined in values
    Args:
        values (tuple): Tuple of values relevant to a given image - created by the train_test_val_instances function

    Returns:
        tuple: Actual modified image, and the target class index for that image
    """
    image_path, target, crop_params = values
    
    # pretransform should always return a tuple of this structure (some image data, some target class index)
    return (_crop(sample, crop_params), target)
    
```

```python
def train_test_val_instances(split, directory, class_to_idx, index, is_valid_file):
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
    for target_class in sorted(class_to_idx.keys()):
        i += 1
        if not os.path.isdir(directory):
            continue
        instances = []
        for filename in index[target_class]:
            if is_valid_file(filename):
                path = os.path.join(directory, filename)
                instances.append((path, class_to_idx[target_class]))

        trainp, _, valp = split

        train += instances[:int(len(instances)*trainp)]
        test += instances[int(len(instances)*trainp):int(len(instances)*(1-valp))]
        val += instances[int(len(instances)*(1-valp)):]
    return train, test, val
```
