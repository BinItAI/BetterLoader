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

## Usage
BetterLoader really shines when you're working with a dataset, and you want to load subsets of image classes conditionally. Say you have 3 folders of images, and you only want to load those images that conform to a specific condition, <b>or</b> those that are present in a pre-defined subset file. What if you want to load a specific set of crops per source image, given a set of source images? BetterLoader can do all this, and more.<br />
<b>Note:</b> BetterLoader currently only supports supervised deep learning tasks. Unsupervised learning support coming soon!

### Basic Usage
Using BetterLoader with its default parameters lets it function just like the regular Python dataloader. A few points worth noting are that:
- BetterLoader does not expect a nested folder structure. In its current iteration, files are expected to all be present in the root directory. This lets us use index files to define classes and labels dynamically, and vary them from experiment to experiment.
- <b>Every</b> instance of BetterLoader requires an index file to function. The index file maps ipmages to their class values. As of now, index files are simply filenames, but we can potentially add support for regex/more complicated logical expressions down the road. Sample index files may be found <a href="/docs/files">here</a>.

A sample use-case for BetterLoader may be found below. It's worth noting that at this point in time, the BetterLoader class has only one callable function.
```python
from betterloader import BetterLoader

index_json = './examples/sample_index.json'
basepath = "./examples/sample_dataset/"
batch_size = 2

loader = BetterLoader(basepath=basepath, index_json_path=index_json)
dataloaders, sizes = loader.fetch_segmented_dataloaders(batch_size=batch_size, transform=None)

print("Dataloader sizes: {}".format(str(sizes)))
```

### Constructor Parameters
| field        |      type      |   description | optional (datatype) |
| ------------- | :-----------: | -----: | -----------: |
| basepath      | str | path to image directory | no |
| index_json_path      | str | path to index file | no |
| num_workers      | int | number of workers | yes (1) |
| subset_json_path      | str | path to subset json file | yes (None) |
| dataset_metadata      |   metadata object for dataset    |   list of optional metadata attributes to customise the BetterLoader | yes (None) |

#### Dataset Metadata
BetterLoader accepts certain key value pairs as dataset metadata, in order to enable some custom functionality.
1. pretransform (callable, optional): This allows us to load a custom pretransform before images are loaded into the dataloader and transformed.
2. classdata (callable, optional): Defines a custom mapping for a custom format index file to read data from the DatasetFolder class.
3. split (tuple, optional): Defines a tuple for train, test, val values which must add to one.
4. train_test_val_instances (callable, optional): Defines a custom function to read values from the index file