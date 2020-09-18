'''Test Suite for BetterLoader
'''
import unittest
from betterloader import BetterLoader
from betterloader.defaults import simple_metadata

# pylint: disable=no-self-use


class Integration(unittest.TestCase):
    """Suite of Integration tests for BetterLoader
    """

    def test_defaults(self):
        '''Test the BetterLoader call using the default parameters
        '''
        index_json = './examples/sample_index.json'
        basepath = "./examples/sample_dataset/"
        batch_size = 2

        loader = BetterLoader(basepath=basepath, index_json_path=index_json)
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=None)

        assert not dataloaders is None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_simple_metadata(self):
        '''Test the BetterLoader call using the same default functions, but passed in this time
        '''
        index_json = './examples/sample_index.json'
        basepath = "./examples/sample_dataset/"
        batch_size = 2

        dataset_metadata = simple_metadata()

        loader = BetterLoader(
            basepath=basepath, index_json_path=index_json, dataset_metadata=dataset_metadata)
        dataloaders, sizes = loader.fetch_segmented_dataloaders(
            batch_size=batch_size, transform=None)

        assert not dataloaders is None

        assert sizes["train"] == 4
        assert sizes["test"] == 2
        assert sizes["val"] == 2

    def test_bad_paths(self):
        '''Test the BetterLoader call using two bad paths - the basepath should be the first exception thrown
        '''
        index_json = './badpath/'
        basepath = "./badpath/"

        dataset_metadata = simple_metadata()
        self.assertRaisesRegex(FileNotFoundError, "Please supply a valid path to your base folder!",
                               BetterLoader, basepath, index_json, dataset_metadata)

    def test_bad_basepath(self):
        '''Test the BetterLoader call using a bad basepath
        '''
        index_json = './examples/sample_index.json'
        basepath = "./badpath/"

        dataset_metadata = simple_metadata()
        self.assertRaisesRegex(FileNotFoundError, "Please supply a valid path to your base folder!",
                               BetterLoader, basepath, index_json, dataset_metadata)

    def test_bad_index(self):
        '''Test the BetterLoader call using a bad index path
        '''
        index_json = './badpath'
        basepath = "./examples/sample_dataset/"

        dataset_metadata = simple_metadata()
        self.assertRaisesRegex(FileNotFoundError, "Please supply a valid path to a dataset index file!",
                               BetterLoader, basepath, index_json, dataset_metadata)


if __name__ == '__main__':
    unittest.main()
