import numpy as np
from torchvision import transforms
np.random.seed(1)

class TransformWhileSampling(object):

    def __init__(self, transform):
        self.transform = transform


    def __call__(self, sample):

        x1 = self.transform(sample)
        x2 = self.transform(sample)

        return x1, x2