---
id: files
title: Index & Subset Files
sidebar_label: Index & Subset Files
slug: /files
---

## Overview
BetterLoader uses two types of files to do some really interesting stuff. These are <a href="#index-files">index files</a> and <a href="#subset-files">subset files</a>.
Index files allow you to specify labelled groupings for your image dataset, which allows you to maintain your actual data within a single flat folder. Subset files, on the other hand, allow you to specify a list of image paths to load, which consequently are labelled via the index file. This allows you to load subsets of your dataset, and run multiple experiments all with minimal file management.

### Index Files
Index JSON files are default used to create a mapping from label, to filenames. Index files are by default, expected to be formatted as key-value pairs, where the values are lists of filenames. However, this format can be overriden by passing a value to the `train_test_val_instances` key of the `dataset_metadata` parameter of the BetterLoader constructor.<br /> Since the format is so flexible there are many things you can do, for example index files can use regex as long as the train_test_val_instances function is setup to parse the regex correctly. There's nothing hardcoded about the index file except that it has to be a json.  A sample index file would look something like:

```json
{
	"class1":["image0.jpg","image1.jpg","image2.jpg","image3.jpg"],
	"class2":["image4.jpg","image5.jpg","image6.jpg","image7.jpg"]
}
```

An index file can also look like the below. The only catch here is that if you do this, you will have to pass a custom function as the `train_test_val_instances` key of the `dataset_metadata` parameter of the BetterLoader constructor. An example of such a function to group filenames based on regex can be found <a href="https://github.com/BinItAI/BetterLoader/tree/master/betterloader/defaults/regex.py">here.</a>
```json
{
	"class1": "^a",
	"class2": "^b"
}
```

### Subset Files
As their names suggest, subset JSON files are used to instruct the BetterLoader to limit itself to only a subset of the dataset present at the root of the directory being loaded from. Currently, subset files just consist of a list of allowed files (as we've been auto-generating them as a part of our workflow), but this is definitely something we would be open to refining as well. A sample subset file would look something like this:
```json
["image0.jpg","image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
```

## Usage
An index is required to use the BetterLoader, either a path to an index json must be supplied, or an index object. The index object may be any python object, and replaces what would be the result of loading the index json. An index is required because it replaces the traditional approach that the PyTorch dataloader uses involving using folder names to infer class label. Since we've done away with this mechanism entirely, an index is essential to loading data for supervised learning tasks.<br />
Subset files, are an optional parameter. If a subset file is not specified, then the BetterLoader will just load your entire dataset :)
You may also use a subset object, which is entirely analogous to the way index objects work.