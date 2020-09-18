'''A collection of aggregated default methods and metadata accessors, used for both testing and default values
'''

from .simple import train_test_val_instances, classdata, pretransform

def simple_metadata():
    """Create a very simple metadata object to test with
    """
    metadata = {}
    metadata["pretransform"] = pretransform
    metadata["classdata"] = classdata
    metadata["train_test_val_instances"] = train_test_val_instances
    return metadata
