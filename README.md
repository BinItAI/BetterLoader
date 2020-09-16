# BetterLoader
> Making it harder to do easy things, but easier to do harder things with the Pytorch Dataloader :)

BetterLoader is an extension of the default PyTorch dataloader class, that allows for custom transformations pre-load and image subset definitions. Use the power of custom index files to maintain only a single copy of a dataset with a fixed, flat file structure, and allow BetterLoader to do all the heavy lifting.

## Installation
```sh
pip install betterloader
```

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
