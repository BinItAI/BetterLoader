(window.webpackJsonp=window.webpackJsonp||[]).push([[15],{73:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return i})),a.d(t,"metadata",(function(){return s})),a.d(t,"rightToc",(function(){return l})),a.d(t,"default",(function(){return c}));var r=a(2),n=a(6),o=(a(0),a(82)),i={id:"gettingstarted",title:"Getting Started",sidebar_label:"Getting Started",slug:"/"},s={unversionedId:"gettingstarted",id:"gettingstarted",isDocsHomePage:!1,title:"Getting Started",description:"Installation",source:"@site/docs/gettingstarted.md",slug:"/",permalink:"/BetterLoader/docs/",editUrl:"https://github.com/binitai/BetterLoader/docs/gettingstarted.md",version:"current",sidebar_label:"Getting Started",sidebar:"someSidebar",next:{title:"Index & Subset Files",permalink:"/BetterLoader/docs/files"}},l=[{value:"Installation",id:"installation",children:[{value:"Python",id:"python",children:[]},{value:"From Source",id:"from-source",children:[]}]},{value:"Usage",id:"usage",children:[{value:"Basic Usage",id:"basic-usage",children:[]},{value:"Constructor Parameters",id:"constructor-parameters",children:[]}]}],b={rightToc:l};function c(e){var t=e.components,a=Object(n.a)(e,["components"]);return Object(o.b)("wrapper",Object(r.a)({},b,a,{components:t,mdxType:"MDXLayout"}),Object(o.b)("h2",{id:"installation"},"Installation"),Object(o.b)("h3",{id:"python"},"Python"),Object(o.b)("p",null,"The BetterLoader library is hosted on ",Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"https://pypi.org/"}),"pypi")," and can be installed via ",Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"https://pip.pypa.io/en/stable/"}),"pip"),"."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-bash"}),"pip install betterloader\n")),Object(o.b)("h3",{id:"from-source"},"From Source"),Object(o.b)("p",null,"For developers, BetterLoader's source may also be found at our ",Object(o.b)("a",Object(r.a)({parentName:"p"},{href:"https://github.com/BinItAI/BetterLoader"}),"Github repository"),". You can also install BetterLoader from source, but if you're just trying to use the package, pip is probably a far better bet."),Object(o.b)("h2",{id:"usage"},"Usage"),Object(o.b)("p",null,"BetterLoader really shines when you're working with a dataset, and you want to load subsets of image classes conditionally. Say you have 3 folders of images, and you only want to load those images that conform to a specific condition, ",Object(o.b)("b",null,"or")," those that are present in a pre-defined subset file. What if you want to load a specific set of crops per source image, given a set of source images? BetterLoader can do all this, and more.",Object(o.b)("br",null)),Object(o.b)("b",null,"Note:")," BetterLoader currently only supports supervised deep learning tasks. Unsupervised learning support coming soon!",Object(o.b)("h3",{id:"basic-usage"},"Basic Usage"),Object(o.b)("p",null,"Using BetterLoader with its default parameters lets it function just like the regular Python dataloader. A few points worth noting are that:"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"BetterLoader does not expect a nested folder structure. In its current iteration, files are expected to all be present in the root directory. This lets us use index files to define classes and labels dynamically, and vary them from experiment to experiment."),Object(o.b)("li",{parentName:"ul"},Object(o.b)("b",null,"Every")," instance of BetterLoader requires an index file to function. The default index file format maps class names to a list of image paths, but the index file can be any json file as long as you modify train_test_val_instances to parse it correctly; for example you could instead map class names to regex for the file paths and pass a train_test_val_instances that reads the files based on that regex. Sample index files may be found ",Object(o.b)("a",{href:"/docs/files"},"here"),".")),Object(o.b)("p",null,"A sample use-case for BetterLoader may be found below. It's worth noting that at this point in time, the BetterLoader class has only one callable function."),Object(o.b)("pre",null,Object(o.b)("code",Object(r.a)({parentName:"pre"},{className:"language-python"}),'from betterloader import BetterLoader\n\nindex_json = \'./examples/sample_index.json\'\nbasepath = "./examples/sample_dataset/"\nbatch_size = 2\n\nloader = BetterLoader(basepath=basepath, index_json_path=index_json)\ndataloaders, sizes = loader.fetch_segmented_dataloaders(batch_size=batch_size, transform=None)\n\nprint("Dataloader sizes: {}".format(str(sizes)))\n')),Object(o.b)("h3",{id:"constructor-parameters"},"Constructor Parameters"),Object(o.b)("table",null,Object(o.b)("thead",{parentName:"table"},Object(o.b)("tr",{parentName:"thead"},Object(o.b)("th",Object(r.a)({parentName:"tr"},{align:null}),"field"),Object(o.b)("th",Object(r.a)({parentName:"tr"},{align:"center"}),"type"),Object(o.b)("th",Object(r.a)({parentName:"tr"},{align:"right"}),"description"),Object(o.b)("th",Object(r.a)({parentName:"tr"},{align:"right"}),"optional (datatype)"))),Object(o.b)("tbody",{parentName:"table"},Object(o.b)("tr",{parentName:"tbody"},Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:null}),"basepath"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"center"}),"str"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"path to image directory"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"no")),Object(o.b)("tr",{parentName:"tbody"},Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:null}),"index_json_path"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"center"}),"str"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"path to index file"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"no")),Object(o.b)("tr",{parentName:"tbody"},Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:null}),"num_workers"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"center"}),"int"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"number of workers"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"yes (1)")),Object(o.b)("tr",{parentName:"tbody"},Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:null}),"subset_json_path"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"center"}),"str"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"path to subset json file"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"yes (None)")),Object(o.b)("tr",{parentName:"tbody"},Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:null}),"dataset_metadata"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"center"}),"metadata object for dataset"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"list of optional metadata attributes to customise the BetterLoader"),Object(o.b)("td",Object(r.a)({parentName:"tr"},{align:"right"}),"yes (None)")))),Object(o.b)("h4",{id:"dataset-metadata"},"Dataset Metadata"),Object(o.b)("p",null,"BetterLoader accepts certain key value pairs as dataset metadata, in order to enable some custom functionality."),Object(o.b)("ol",null,Object(o.b)("li",{parentName:"ol"},"pretransform (callable, optional): This allows us to load a custom pretransform before images are loaded into the dataloader and transformed."),Object(o.b)("li",{parentName:"ol"},"classdata (callable, optional): Defines a custom mapping for a custom format index file to read data from the DatasetFolder class."),Object(o.b)("li",{parentName:"ol"},"split (tuple, optional): Defines a tuple for train, test, val values which must add to one."),Object(o.b)("li",{parentName:"ol"},"train_test_val_instances (callable, optional): Defines a custom function to read values from the index file")))}c.isMDXComponent=!0}}]);