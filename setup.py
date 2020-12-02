"""setup file for BetterLoader deployments
"""

from os import path
from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="BetterLoader",
    version="0.2.0",
    description="A better PyTorch dataloader",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BinItAI/BetterLoader",
    author="BinIt Inc",
    author_email="support@binit.in",
    license="MIT",
    download_url="https://github.com/BinItAI/BetterLoader/archive/0.1.5.zip",
    packages=["betterloader"],
    install_requires=[
        "future==0.18.2",
        "numpy==1.19.1",
        "Pillow==7.2.0",
        "torch==1.7.0",
        "torchvision==0.8.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Programming Language :: Python :: 3.7",
    ],
)
