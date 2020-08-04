from torchvision.datasets import VisionDataset
import numpy as np

from PIL import Image

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

#def cat_from_path(path):
#    return path[path.rfind('_',0,path.rfind('_')-1)+1:path.rfind('_')]

def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None,instance='train',index=None,train_test_val_instances = None):
    #instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    
    train,test,val = train_test_val_instances(directory,class_to_idx,index)

    # train, test, val = [], [], []
    # #import glob
    # i = 0

    
    # for target_class in sorted(class_to_idx.keys()):
    #     #print(i)
    #     i+=1
        
    #     class_index = class_to_idx[target_class]
    #     target_dir = directory#os.path.join(directory), target_class)
    #     if not os.path.isdir(target_dir):
    #         continue
    #     instances = []
    #     '''
    #     for file in index[target_class]:
    #         #if file[file.rfind('_')+1:file.rfind('.')] == target_class:
    #         if is_valid_file(file):
    #             for pt in pts[file]:
    #                 path = os.path.join(target_dir,file)
    #                 item = (path, class_index, pt)
    #                 instances.append(item)
    #     '''
    #     for it in index[target_class]:
    #         instances += [(target_dir +'/'+ it[0]+'.jpg', class_index, (it[1],it[2]))]

        
        
    #     trainp = 0.6
    #     testp = 0.2
    #     valp = 0.2

    #     train += instances[:int(len(instances)*trainp)]
    #     test += instances[int(len(instances)*trainp):int(len(instances)*(1-valp))]
    #     val += instances[int(len(instances)*(1-valp)):]
    
        '''
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)'''
    if instance == 'train':
        return train
    if instance == 'test':
        return test
    else:
        return val
    #return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, instance='train', index = None, train_test_val_instances=None, class_data=None,pretransform = None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.class_data = class_data
        self.pretransform = pretransform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file, instance, index, train_test_val_instances)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        # classes =list(set([cat_from_path(d) for d in os.listdir(dir)]))
        # #print(classes)
        # classes = [str(x) for x in range(1,23)]
        # classes.sort()
        # class_to_idx = {classes[i]: i for i in range(len(classes))}
        classes,class_to_idx = self.class_data

        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # path, target, pt = self.samples[index]


        values = self.samples[index]
        path = values[0]
        sample = self.loader(path)
        sample, target = self.pretransform(sample,values)
        # w,h = sample.size[:2]
        # pt0 = float(pt[0]) * w
        # pt1 = float(pt[1]) * h
        # d = np.min([w,h])
        # #print(d)
        # p = int(256./1100. * d)
        # #print(p)
        # p = int(p/1.414)
        # #print(sample.size)
        # #print(p)
        # ps = (pt0-p,pt1-p,pt0+p,pt1+p)
        # ps_m = (0,0,w,h)
        # ps2 = [int(np.min([x,y])) if y != 0 else int(np.max([x,y])) for x,y in zip(ps,ps_m)]
        # #print(ps2)
        # sample = sample.crop(ps2)

        

        #print(sample.size)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderCustom(DatasetFolder):
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
                 loader=default_loader, is_valid_file=None, instance = 'train',index = None, train_test_val_instances=None, class_data=None, pretransform = None):
        super(ImageFolderCustom, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, 
                                          instance = instance,
                                          index = index,
                                          train_test_val_instances = train_test_val_instances,
                                          class_data=class_data,
                                          pretransform = pretransform)
        self.imgs = self.samples
