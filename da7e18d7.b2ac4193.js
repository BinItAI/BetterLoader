(window.webpackJsonp=window.webpackJsonp||[]).push([[13],{73:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return s})),a.d(t,"metadata",(function(){return o})),a.d(t,"rightToc",(function(){return l})),a.d(t,"default",(function(){return d}));var n=a(2),r=a(6),i=(a(0),a(82)),s={id:"gettingstarted",title:"Getting Started",sidebar_label:"Getting Started",slug:"/"},o={unversionedId:"gettingstarted",id:"gettingstarted",isDocsHomePage:!1,title:"Getting Started",description:"Installation",source:"@site/docs/gettingstarted.md",slug:"/",permalink:"/BetterLoader/docs/",editUrl:"https://github.com/binitai/BetterLoader/docs/gettingstarted.md",version:"current",sidebar_label:"Getting Started",sidebar:"someSidebar",next:{title:"Index & Subset Files",permalink:"/BetterLoader/docs/files"}},l=[{value:"Installation",id:"installation",children:[{value:"Python",id:"python",children:[]},{value:"From Source",id:"from-source",children:[]}]},{value:"Why BetterLoader?",id:"why-betterloader",children:[{value:"Creating a BetterLoader",id:"creating-a-betterloader",children:[]},{value:"Constructor Parameters",id:"constructor-parameters",children:[]},{value:"Usage",id:"usage",children:[]}]}],c={rightToc:l};function d(e){var t=e.components,a=Object(r.a)(e,["components"]);return Object(i.b)("wrapper",Object(n.a)({},c,a,{components:t,mdxType:"MDXLayout"}),Object(i.b)("h2",{id:"installation"},"Installation"),Object(i.b)("h3",{id:"python"},"Python"),Object(i.b)("p",null,"The BetterLoader library is hosted on ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://pypi.org/"}),"pypi")," and can be installed via ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://pip.pypa.io/en/stable/"}),"pip"),"."),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"pip install betterloader\n")),Object(i.b)("h3",{id:"from-source"},"From Source"),Object(i.b)("p",null,"For developers, BetterLoader's source may also be found at our ",Object(i.b)("a",Object(n.a)({parentName:"p"},{href:"https://github.com/BinItAI/BetterLoader"}),"Github repository"),". You can also install BetterLoader from source, but if you're just trying to use the package, pip is probably a far better bet."),Object(i.b)("h2",{id:"why-betterloader"},"Why BetterLoader?"),Object(i.b)("p",null,"BetterLoader really shines when you're working with a dataset, and you want to load subsets of image classes conditionally. Say you have 3 folders of images, and you only want to load those images that conform to a specific condition, ",Object(i.b)("b",null,"or")," those that are present in a pre-defined subset file. What if you want to load a specific set of crops per source image, given a set of source images? BetterLoader can do all this, and more.",Object(i.b)("br",null)),Object(i.b)("b",null,"Note:")," BetterLoader currently only supports supervised deep learning tasks. Unsupervised learning support coming soon!",Object(i.b)("h3",{id:"creating-a-betterloader"},"Creating a BetterLoader"),Object(i.b)("p",null,"Using BetterLoader with its default parameters lets it function just like the regular Python dataloader. A few points worth noting are that:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"BetterLoader does not expect a nested folder structure. In its current iteration, files are expected to all be present in the root directory. This lets us use index files to define classes and labels dynamically, and vary them from experiment to experiment."),Object(i.b)("li",{parentName:"ul"},Object(i.b)("b",null,"Every")," instance of BetterLoader requires an index file to function. The default index file format maps class names to a list of image paths, but the index file can be any json file as long as you modify train_test_val_instances to parse it correctly; for example you could instead map class names to regex for the file paths and pass a train_test_val_instances that reads the files based on that regex. Sample index files may be found ",Object(i.b)("a",{href:"/docs/files"},"here"),".")),Object(i.b)("p",null,"A sample use-case for BetterLoader may be found below. It's worth noting that at this point in time, the BetterLoader class has only one callable function."),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'from betterloader import BetterLoader\n\nindex_json = \'./examples/sample_index.json\'\n# or index_object = {"class1":["image0.jpg","image1.jpg","image2.jpg","image3.jpg"],"class2":["image4.jpg","image5.jpg","image6.jpg","image7.jpg"]}\nbasepath = "./examples/sample_dataset/"\nbatch_size = 2\n\nloader = BetterLoader(basepath=basepath, index_json_path=index_json)\n# or loader = BetterLoader(basepath=basepath, index_object=index_object)\ndataloaders, sizes = loader.fetch_segmented_dataloaders(batch_size=batch_size, transform=None)\n\nprint("Dataloader sizes: {}".format(str(sizes)))\n')),Object(i.b)("h3",{id:"constructor-parameters"},"Constructor Parameters"),Object(i.b)("table",null,Object(i.b)("thead",{parentName:"table"},Object(i.b)("tr",{parentName:"thead"},Object(i.b)("th",Object(n.a)({parentName:"tr"},{align:null}),"field"),Object(i.b)("th",Object(n.a)({parentName:"tr"},{align:"center"}),"type"),Object(i.b)("th",Object(n.a)({parentName:"tr"},{align:"right"}),"description"),Object(i.b)("th",Object(n.a)({parentName:"tr"},{align:"right"}),"optional (datatype)"))),Object(i.b)("tbody",{parentName:"table"},Object(i.b)("tr",{parentName:"tbody"},Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:null}),"basepath"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"center"}),"str"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"path to image directory"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"no")),Object(i.b)("tr",{parentName:"tbody"},Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:null}),"index_json_path"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"center"}),"str"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"path to index file"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"yes (None)")),Object(i.b)("tr",{parentName:"tbody"},Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:null}),"index_object"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"center"}),"dict"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"An object representation of an index file"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"yes (None)")),Object(i.b)("tr",{parentName:"tbody"},Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:null}),"num_workers"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"center"}),"int"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"number of workers"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"yes (1)")),Object(i.b)("tr",{parentName:"tbody"},Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:null}),"subset_json_path"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"center"}),"str"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"path to subset json file"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"yes (None)")),Object(i.b)("tr",{parentName:"tbody"},Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:null}),"subset_object"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"center"}),"dict"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"An object representation of the subset file"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"yes (None)")),Object(i.b)("tr",{parentName:"tbody"},Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:null}),"dataset_metadata"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"center"}),"metadata object for dataset"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"list of optional metadata attributes to customise the BetterLoader"),Object(i.b)("td",Object(n.a)({parentName:"tr"},{align:"right"}),"yes (None)")))),Object(i.b)("h3",{id:"usage"},"Usage"),Object(i.b)("p",null,"The BetterLoader class' ",Object(i.b)("inlineCode",{parentName:"p"},"fetch_segmented_dataloaders")," function allows for a user to obtain a tuple of dictionaries, which are most commonly referenced as ",Object(i.b)("inlineCode",{parentName:"p"},"(dataloaders, sizes)"),". Each dictionary consequently contains ",Object(i.b)("inlineCode",{parentName:"p"},"train"),", ",Object(i.b)("inlineCode",{parentName:"p"},"test"),", and ",Object(i.b)("inlineCode",{parentName:"p"},"val")," keys, allowing for easy access to the dataloaders, as well as their sizes. The function header for the same may be found below:"),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{}),"def fetch_segmented_dataloaders(self, batch_size, transform=None)\n\"\"\"Fetch custom dataloaders, which may be used with any PyTorch model\n    Args:\n    batch_size (string): Image batch size.\n    transform (callable or dict, optional): PyTorch transform object. This parameter may also be a dict with keys of 'train', 'test', and 'val', in order to enable separate transforms for each split.\n    Returns:\n        dict: A dictionary of dataloaders for train test split\n        dict: A dictionary of dataset sizes for train test split\n\"\"\"\n")),Object(i.b)("h4",{id:"dataset-metadata"},"Dataset Metadata"),Object(i.b)("p",null,"BetterLoader accepts certain key value pairs as dataset metadata, in order to enable some custom functionality."),Object(i.b)("ol",null,Object(i.b)("li",{parentName:"ol"},"pretransform (callable, optional): This allows us to load a custom pretransform before images are loaded into the dataloader and transformed.\nFor basic usage a pretransform that does not do anything (the default) is usually sufficient. An example use case for the customizability is listed below."),Object(i.b)("li",{parentName:"ol"},"classdata (callable, optional): Defines a custom mapping for a custom format index file to read data from the DatasetFolder class.\nSince the index file may have any structure we need to ensure that the classes and a mapping from the classes to the index are always available.\nReturns a tuple (list of classes, dictionary mapping of class to index)"),Object(i.b)("li",{parentName:"ol"},"split (tuple, optional): Defines a tuple for train, test, val values which must add to one."),Object(i.b)("li",{parentName:"ol"},"train_test_val_instances (callable, optional): Defines a custom function to read values from the index file.\nThe default expects an index that is a dict mapping classes to a list of file paths, will need to be written custom for different index formats.\nAlways must return train test and val splits, which each need to be a list of tuples, each tuple corresponding to one datapoint.\nThe first element of this tuple must also be the filepath of the image for that datapoint.\nThe default also has the target class index as the second element of this tuple, this is probably good for most use cases.\nEach of these datapoint tuples is passed as the ",Object(i.b)("inlineCode",{parentName:"li"},"values")," argument in the pretransform, any additional data necessary for transforming the datapoint before it is loaded can go in the datapoint tuple."),Object(i.b)("li",{parentName:"ol"},"supervised (bool, optional): Defines whether or not the experiment is supervised"),Object(i.b)("li",{parentName:"ol"},"custom_collator (callable, optional): Custom function that merges a list of samples to form a mini-batch of Tensors"),Object(i.b)("li",{parentName:"ol"},"drop_last (bool, optional): Defines whether to drop the last incomplete batch if the dataset is not divisible by batch size to avoid sizing errors"),Object(i.b)("li",{parentName:"ol"},"pin_mem (bool, optional): Sets the data load to copy tensors into CUDA pinned memory before returning them, providing your data elements are not custom type"),Object(i.b)("li",{parentName:"ol"},"sampler (torch.utils.data.Sampler or ",Object(i.b)("inlineCode",{parentName:"li"},"iterable"),", optional): Can be used to define a custom strategy to draw data from the dataset")),Object(i.b)("hr",null),Object(i.b)("p",null,"Here is an example of a ",Object(i.b)("inlineCode",{parentName:"p"},"pretransform")," and a ",Object(i.b)("inlineCode",{parentName:"p"},"train_test_val_instances")," designed to allow for a specified crop to be taken of each image."),Object(i.b)("b",null,"Notes"),":",Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"The internals of the loader dictate that the elements of the ",Object(i.b)("inlineCode",{parentName:"li"},"instances")," variables generated from train_test_val_instances will become the ",Object(i.b)("inlineCode",{parentName:"li"},"values")," argument for a pretransform call, and the ",Object(i.b)("inlineCode",{parentName:"li"},"sample")," argument for pretransform is the image data loaded directly from the filepath in ",Object(i.b)("inlineCode",{parentName:"li"},"values[0]")," (or ",Object(i.b)("inlineCode",{parentName:"li"},"instances[i][0]"),")."),Object(i.b)("li",{parentName:"ul"},"Since the index file here has a similar structure to the default we can get away with using the default classdata function, but index files that don't have the classes as keys of a dictionary will need a custom way of determining the classes.")),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'def pretransform(sample, values):\n    """Example pretransform - takes an image and crops it based on the parameters defined in values\n    Args:\n        values (tuple): Tuple of values relevant to a given image - created by the train_test_val_instances function\n\n    Returns:\n        tuple: Actual modified image, and the target class index for that image\n    """\n    image_path, target, crop_params = values\n    \n    # pretransform should always return a tuple of this structure (some image data, some target class index)\n    return (_crop(sample, crop_params), target)\n    \n')),Object(i.b)("pre",null,Object(i.b)("code",Object(n.a)({parentName:"pre"},{className:"language-python"}),'def train_test_val_instances(split, directory, class_to_idx, index, is_valid_file):\n    """Function to perform default train/test/val instance creation\n    Args:\n        split (tuple): Tuple of ratios (from 0 to 1) for train, test, val values\n        directory (str): Parent directory to read images from\n        class_to_idx (dict): Dictionary to map values from class strings to index values\n        index (dict): Index file dict object\n        is_valid_file (callable): Function to verify if a file should be loaded\n    Returns:\n        (tuple): Tuple of length 3 containing train, test, val instances\n    """\n    train, test, val = [], [], []\n    i = 0\n    for target_class in sorted(class_to_idx.keys()):\n        i += 1\n        if not os.path.isdir(directory):\n            continue\n        instances = []\n        for filename in index[target_class]:\n            if is_valid_file(filename):\n                path = os.path.join(directory, filename)\n                instances.append((path, class_to_idx[target_class]))\n\n        trainp, _, valp = split\n\n        train += instances[:int(len(instances)*trainp)]\n        test += instances[int(len(instances)*trainp):int(len(instances)*(1-valp))]\n        val += instances[int(len(instances)*(1-valp)):]\n    return train, test, val\n')))}d.isMDXComponent=!0}}]);