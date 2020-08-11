# BetterLoader
> Making it harder to do easy things, but easier to do harder things with the Pytorch Dataloader :)

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

BetterLoader is an extension of the default PyTorch dataloader class, that allows for custom transformations pre-load and image subset definitions. Use the power of custom index files to maintain only a single copy of a dataset with a fixed, flat file structure, and allow BetterLoader to do all the heavy lifting.

## Installation

### Pip
```sh
pip install betterloader
```

## Usage example
Check out our <a href="./examples">examples</a> folder. The BetterLoader wiki is coming soon!

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

## Release History
* 0.0.1
    * Work in progress

## Meta
Distributed under the MIT license. See ``LICENSE`` for more information.

## Todo
There are a ton of open issues! Do check them out if you would like to contribute

## Contributing
1. Fork this repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request, and we'll merge it in if it works :)