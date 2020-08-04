from setuptools import setup

setup(
    name='BetterLoader',
    version='0.1.0',    
    description='A better PyTorch dataloader',
    url='https://github.com/BinItAI/BetterLoader',
    author='BinIt Inc',
    author_email='',
    license='MIT',
    packages=['betterloader'],
    install_requires=['torchvision>0.4.2','numpy'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Python :: 3.7',
    ],
)