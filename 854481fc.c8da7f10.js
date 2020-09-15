(window.webpackJsonp=window.webpackJsonp||[]).push([[7],{66:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return n})),a.d(t,"metadata",(function(){return o})),a.d(t,"rightToc",(function(){return r})),a.d(t,"default",(function(){return u}));var i=a(2),s=a(6),l=(a(0),a(82)),n={id:"files",title:"Index & Subset Files",sidebar_label:"Index & Subset Files",slug:"/files"},o={unversionedId:"files",id:"files",isDocsHomePage:!1,title:"Index & Subset Files",description:"Overview",source:"@site/docs/files.md",slug:"/files",permalink:"/betterloader/docs/files",editUrl:"https://github.com/binitai/betterloader/docs/files.md",version:"current",sidebar_label:"Index & Subset Files",sidebar:"someSidebar",previous:{title:"Getting Started",permalink:"/betterloader/docs/"}},r=[{value:"Overview",id:"overview",children:[{value:"Index Files",id:"index-files",children:[]},{value:"Subset Files",id:"subset-files",children:[]}]},{value:"Usage",id:"usage",children:[]}],d={rightToc:r};function u(e){var t=e.components,a=Object(s.a)(e,["components"]);return Object(l.b)("wrapper",Object(i.a)({},d,a,{components:t,mdxType:"MDXLayout"}),Object(l.b)("h2",{id:"overview"},"Overview"),Object(l.b)("p",null,"BetterLoader uses two types of files to do some really interesting stuff. These are ",Object(l.b)("a",{href:"#index-files"},"index files")," and ",Object(l.b)("a",{href:"#subset-files"},"subset files"),".\nIndex files allow you to specify labelled groupings for your image dataset, which allows you to maintain your actual data within a single flat folder. Subset files, on the other hand, allow you to specify a list of image paths to load, which consequently are labelled via the index file. This allows you to load subsets of your dataset, and run multiple experiments all with minimal file management."),Object(l.b)("h3",{id:"index-files"},"Index Files"),Object(l.b)("p",null,"Index JSON files are default used to create a mapping from label, to filenames. Index files are by default, expected to be formatted as key-value pairs, where the values are lists of filenames. However, this format can be overriden by passing a value to the ",Object(l.b)("inlineCode",{parentName:"p"},"train_test_val_instances")," key of the ",Object(l.b)("inlineCode",{parentName:"p"},"dataset_metadata")," parameter of the BetterLoader constructor.",Object(l.b)("br",null)," Since the format is so flexible there are many things you can do, for example index files can use regex as long as the train_test_val_instances function is setup to parse the regex correctly. There's nothing hardcoded about the index file except that it has to be a json.  A sample index file would look something like:"),Object(l.b)("pre",null,Object(l.b)("code",Object(i.a)({parentName:"pre"},{className:"language-json"}),'{\n    "class1":["image0.jpg","image1.jpg","image2.jpg","image3.jpg"],\n    "class2":["image4.jpg","image5.jpg","image6.jpg","image7.jpg"]\n}\n')),Object(l.b)("h3",{id:"subset-files"},"Subset Files"),Object(l.b)("p",null,"As their names suggest, subset JSON files are used to instruct the BetterLoader to limit itself to only a subset of the dataset present at the root of the directory being loaded from. Currently, subset files just consist of a list of allowed files (as we've been auto-generating them as a part of our workflow), but this is definitely something we would be open to refining as well. A sample subset file would look something like this:"),Object(l.b)("pre",null,Object(l.b)("code",Object(i.a)({parentName:"pre"},{className:"language-json"}),'["image0.jpg","image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]\n')),Object(l.b)("h2",{id:"usage"},"Usage"),Object(l.b)("p",null,"Index files are a ",Object(l.b)("b",null,"required")," parameter for the BetterLoader. This is because they replace the traditional approach that the PyTorch dataloader uses involving using folder names to infer class label. Since we've done away with this mechanism entirely, index files are essential to loading data for supervised learning tasks.",Object(l.b)("br",null),"\nSubset files, are an optional parameter. If a subset file is not specified, then the BetterLoader will just load your entire dataset :)"))}u.isMDXComponent=!0}}]);