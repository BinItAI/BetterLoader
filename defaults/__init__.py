'''
Export predefined detaults for us to use in testing
'''

from .simple_defaults import simple_train_test_val_instances, simple_classdata, simple_pretransform
from .default_collate import basic_collate_fn

def simple_metadata():
    """Create a very simple metadata object to test with
    """
    metadata = {}
    metadata["pretransform"] = simple_pretransform
    metadata["classdata"] = simple_classdata
    metadata["train_test_val_instances"] = simple_train_test_val_instances
    metadata["dataloader_params"] = {'supervised': True}
    return metadata

def complex_metadata():

    metadata = {}
    metadata["pretransform"] = simple_pretransform
    metadata["classdata"] = simple_classdata
    metadata["train_test_val_instances"] = simple_train_test_val_instances
    metadata["dataloader_params"] = {'supervised': True, 'custom_collate': basic_collate_fn, 'drop_last': True,
                                     'eccentric_object':False, 'sample_type': None}

    return metadata


