'''A collection of aggregated default methods and metadata accessors, used for both testing and default values
'''

def simple_metadata():
    """Create a very simple metadata object to test with
    """
    from .simple import train_test_val_instances, classdata, pretransform
    metadata = {}
    metadata["pretransform"] = pretransform
    metadata["classdata"] = classdata
    metadata["train_test_val_instances"] = train_test_val_instances
    return metadata

def regex_metadata():
    """Create a regex based metadata object
    """
    from .regex import train_test_val_instances, classdata, pretransform
    metadata = {}
    metadata = {}
    metadata["pretransform"] = pretransform
    metadata["classdata"] = classdata
    metadata["train_test_val_instances"] = train_test_val_instances
    return metadata

def complex_metadata():
    """Create a collation based metadata object
    """
    from .collate import basic_collate_fn
    from .simple import train_test_val_instances, classdata, pretransform
    metadata = {}
    metadata["pretransform"] = pretransform
    metadata["classdata"] = classdata
    metadata["train_test_val_instances"] = train_test_val_instances
    metadata["dataloader_params"] = {'supervised': True, 'custom_collate': basic_collate_fn, 'drop_last': True,
                                     'eccentric_object':False, 'sample_type': None}

    return metadata
