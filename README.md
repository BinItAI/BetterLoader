# BetterLoader
> Making it harder to do easy things, but easier to do harder things with the Pytorch Dataloader :)

BetterLoader is an extension of the default PyTorch dataloader class, that allows for custom transformations pre-load and image subset definitions. Use the power of custom index files to maintain only a single copy of a dataset with a fixed, flat file structure, and allow BetterLoader to do all the heavy lifting.

## Installation
```sh
pip install betterloader
```

## Usage
BetterLoader allows you to dynamically assign images to labels, load subsets of images conditionally, perform custom pretransforms before loading an image, and much more. 

### Basic Usage
A few points worth noting are that:
- BetterLoader does not expect a nested folder structure. In its current iteration, files are expected to all be present in the root directory.
- <b>Every</b> instance of BetterLoader requires an index file to function. Sample index files may be found <a href="https://binitai.github.io/BetterLoader/docs/files">here</a>.

```python
from betterloader import BetterLoader

index_json = './examples/sample_index.json'
basepath = "./examples/sample_dataset/"
batch_size = 2

loader = BetterLoader(basepath=basepath, index_json_path=index_json)
dataloaders, sizes = loader.fetch_segmented_dataloaders(batch_size=batch_size, transform=None)

print("Dataloader sizes: {}".format(str(sizes)))
```
For more information and more detailed examples, please check out <a href="https://binitai.github.io/BetterLoader/">the BetterLoader docs</a>!

## Development setup

We use <a href="https://opensource.com/article/18/8/what-how-makefile">Makefile</a> to make our lives a little easier :)
### Install Dependancies
```sh
make install
```
### Run Sample
```sh
make sample
```
### Run Unit Tests
```sh
make test
```

## Meta
Distributed under the MIT license. See ``LICENSE`` for more information.

## Documentation & Usage
- [Usage docs](https://binitai.github.io/BetterLoader/)
- [Example implementation](./examples)
- [Contributing](./CONTRIBUTING.md)
