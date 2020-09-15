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
Index JSON files are used to create a mapping from label, to filenames. Index files are by default, expected to be formatted as key-value pairs, where the values are lists of filenames. However, this format can be overriden by passing a value to the `train_test_val_instances` key of the `dataset_metadata` parameter of the BetterLoader constructor.<br />Index files currently only support absolute string comparisons, but this could also be modified to add support for regular expressions. A sample index file would look something like:

```json
{
	"class1":["image0.jpg","image1.jpg","image2.jpg","image3.jpg"],
	"class2":["image4.jpg","image5.jpg","image6.jpg","image7.jpg"]
}
```

### Subset Files
As their names suggest, subset JSON files are used to instruct the BetterLoader to limit itself to only a subset of the dataset present at the root of the directory being loaded from. Currently, subset files just consist of a list of allowed files (as we've been auto-generating them as a part of our workflow), but this is definitely something we would be open to refining as well. A sample subset file would look something like this:
```json
["image0.jpg","image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
```

## Usage
Index files are a <b>required</b> parameter for the BetterLoader. This is because they replace the traditional approach that the PyTorch dataloader uses involving using folder names to infer class label. Since we've done away with this mechanism entirely, index files are essential to loading data for supervised learning tasks.<br />
Subset files, are an optional parameter. If a subset file is not specified, then the BetterLoader will just load your entire dataset :)