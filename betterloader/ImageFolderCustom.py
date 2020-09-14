"""
Modified version of the PyTorch ImageFolder class to make custom dataloading possible

"""

from PIL import Image

from .DatasetFolder import DatasetFolder

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    """open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    Args:
        path: Load image at path
    Returns:
        Pil.Image: A PIL image object in RGB format
    """ 
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    """Helper to try and load image as an accimage, if supported
    Args:
        path: Path to target image
    Returns:
        var: Image object, either as an accimage, or a PIL image
    """
    import accimage #pylint: disable=import-error
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """Load images given a path, either via accimage or PIL
    Args:
        path: Path to target image
    Returns:
        var: Image object, either as an accimage, or a PIL image
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    
    return pil_loader(path)

def default_classdata(_, index):
    """Load default classdata if no class data is passed
    Args:
        _: Ignored path value
        index: Index file dictionary
    Returns:
        classes: A list of image classes
        class_to_idx: Mapping from classes to indexes of those classes
    """
    classes = list(index.keys())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class ImageFolderCustom(DatasetFolder): # pylint: disable=too-few-public-methods
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, instance='train', 
                 index = None, train_test_val_instances=None, class_data=None,
                 pretransform = None):

        class_data = default_classdata if class_data is None else class_data

        super(ImageFolderCustom, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform= transform,
                                          target_transform= target_transform,
                                          is_valid_file= is_valid_file, 
                                          instance = instance,
                                          index = index,
                                          train_test_val_instances = train_test_val_instances,
                                          class_data= class_data,
                                          pretransform = pretransform)
        self.imgs = self.samples
