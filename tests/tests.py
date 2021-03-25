"""Test Suite for BetterLoader
"""
import unittest

import sys
import os

sys.path.append(os.getcwd())

from betterloader import BetterLoader, UnsupervisedBetterLoader
from betterloader.defaults import simple_metadata, regex_metadata, collate_metadata

from torchvision import transforms

# pylint: disable=no-self-use

basic_transform = transforms.Compose(
    [
        transforms.CenterCrop(10),
    ]
)

dictionary_transform = {
    "train": basic_transform,
    "test": basic_transform,
    "val": basic_transform,
}


class Integration(unittest.TestCase):
    """Suite of Integration tests for BetterLoader"""

    def test_defaults(self):
        """Test the BetterLoader call using the default parameters"""
        index_json = "./examples/sample_index.json"
        basepath = "./examples/sample_dataset/"
        batch_size = 2

        loader = BetterLoader(basepath=basepath, index_json_path=index_json)
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=None
        )

        assert dataloaders is not None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_transform(self):
        """Test the BetterLoader call using the default parameters"""
        index_json = "./examples/sample_index.json"
        basepath = "./examples/sample_dataset/"
        batch_size = 2

        loader = BetterLoader(basepath=basepath, index_json_path=index_json)
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=basic_transform
        )

        assert dataloaders is not None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_transformdict(self):
        """Test the BetterLoader call using the default parameters"""
        index_json = "./examples/sample_index.json"
        basepath = "./examples/sample_dataset/"
        batch_size = 2

        loader = BetterLoader(basepath=basepath, index_json_path=index_json)
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=dictionary_transform
        )

        assert dataloaders is not None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_unsupervised(self):
        """Test the BetterLoader Unsupervised call using the default parameters"""
        index_json = "./examples/sample_index_unsupervised.json"
        basepath = "./examples/sample_dataset/"
        batch_size = 2
        metadata = collate_metadata()
        better_loader = UnsupervisedBetterLoader(
            basepath=basepath,
            base_experiment_details=["simclr", 1, (150, 150)],
            index_json_path=index_json,
            dataset_metadata=metadata,
        )
        dataloaders, sizes = better_loader.fetch_segmented_dataloaders(
            batch_size=batch_size
        )

        assert dataloaders is not None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_defaults_with_object(self):
        """Test the BetterLoader call using the default parameters but using index_object not index_json_path"""
        index = {
            "class1": ["image0.jpg", "image1.jpg", "image2.jpg", "image3.jpg"],
            "class2": ["image4.jpg", "image5.jpg", "image6.jpg", "image7.jpg"],
        }
        basepath = "./examples/sample_dataset/"
        batch_size = 2

        loader = BetterLoader(basepath=basepath, index_object=index)
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=None
        )

        assert dataloaders is not None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_simple_metadata(self):
        """Test the BetterLoader call using the same default functions, but passed in this time"""
        index_json = "./examples/sample_index.json"
        basepath = "./examples/sample_dataset/"
        batch_size = 2

        dataset_metadata = simple_metadata()

        loader = BetterLoader(
            basepath=basepath,
            index_json_path=index_json,
            dataset_metadata=dataset_metadata,
        )
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=None
        )

        assert dataloaders is not None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_complex_metadata(self):

        index_json = "./examples/sample_index.json"
        basepath = "./examples/sample_dataset/"
        batch_size = 2

        dataset_metadata = collate_metadata()
        loader = BetterLoader(
            basepath=basepath,
            index_json_path=index_json,
            dataset_metadata=dataset_metadata,
        )
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=None
        )

        assert dataloaders is not None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_regex_metadata(self):
        """Test the BetterLoader call using regex based functions, but passed in this time"""
        index_json = "./examples/regex_index.json"
        basepath = "./examples/regex_dataset/"
        batch_size = 2

        dataset_metadata = regex_metadata()

        loader = BetterLoader(
            basepath=basepath,
            index_json_path=index_json,
            dataset_metadata=dataset_metadata,
        )
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=None
        )

        assert dataloaders is not None

        assert sizes["train"] == 2
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_bad_paths(self):
        """Test the BetterLoader call using two bad paths - the basepath should be the first exception thrown"""
        index_json = "./badpath/"
        basepath = "./badpath/"

        dataset_metadata = simple_metadata()
        self.assertRaisesRegex(
            FileNotFoundError,
            "Please supply a valid path to your base folder!",
            BetterLoader,
            basepath,
            index_json,
            dataset_metadata,
        )

    def test_bad_basepath(self):
        """Test the BetterLoader call using a bad basepath"""
        index_json = "./examples/sample_index.json"
        basepath = "./badpath/"

        dataset_metadata = simple_metadata()
        self.assertRaisesRegex(
            FileNotFoundError,
            "Please supply a valid path to your base folder!",
            BetterLoader,
            basepath,
            index_json,
            dataset_metadata,
        )

    def test_bad_index(self):
        """Test the BetterLoader call using a bad index path"""
        index_json = "./badpath"
        basepath = "./examples/sample_dataset/"

        dataset_metadata = simple_metadata()
        self.assertRaisesRegex(
            FileNotFoundError,
            "Please supply a valid path to a dataset index file or valid index object!",
            BetterLoader,
            basepath,
            index_json,
            dataset_metadata,
        )


if __name__ == "__main__":
    unittest.main()
