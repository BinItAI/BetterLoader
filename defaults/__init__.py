'''
Export predefined detaults for us to use in testing
'''

from .simple_defaults import simple_train_test_val_instances, simple_classdata, simple_pretransform

def simple_metadata():
    """Create a very simple metadata object to test with
    """
    metadata = {}
    metadata["pretransform"] = simple_pretransform
    metadata["classdata"] = simple_classdata
    metadata["train_test_val_instances"] = simple_train_test_val_instances
    return metadata
