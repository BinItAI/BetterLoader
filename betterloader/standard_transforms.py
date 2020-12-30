import cv2
import numpy as np
from torchvision import transforms

np.random.seed(1)

class GaussianBlur(object):

    def __init__(self, kernel_size, min_var=0.1, max_var=2.0):

        self.min_sig = min_var
        self.max_sig = max_var
        self.kernel_size_w = kernel_size[0]
        self.kernel_size_h = kernel_size[1]

    def __call__(self, sample):

        sample = np.array(sample)
        prob = np.random.random_sample()

        if prob < 0.5:

            final_sig = (self.max_sig - self.min_sig) * np.random.random_sample() + self.min_sig
            sample = cv2.GaussianBlur(sample, (self.kernel_size_w, self.kernel_size_h), final_sig)

        return sample


class AverageBlur(object):

    def __init__(self, kernel_size):

        self.kernel_size_w = kernel_size[0]
        self.kernel_size_h = kernel_size[1]

    def __call__(self, sample):

        sample = np.array(sample)
        prob = np.random.random_sample()

        if prob < 0.5:

            sample = cv2.blur(sample, (self.kernel_size_w, self.kernel_size_h))

        return sample


class BilateralBlur(object):

    def __init__(self, kernel_size, min_var_c=0.1, max_var_c=2.0, min_var_s=0.1, max_var_s=2.0):

        self.min_sig_c = min_var_c
        self.max_sig_c = max_var_c
        self.min_sig_s = min_var_s
        self.max_sig_s = max_var_s

        self.kernel_size_w = kernel_size[0]
        self.kernel_size_h = kernel_size[1]

    def __call__(self, sample):

        sample = np.array(sample)
        prob = np.random.random_sample()

        if prob < 0.5:

            final_sig_c = (self.max_sig_c - self.min_sig_c) * np.random.random_sample() + self.min_sig_c
            final_sig_s = (self.max_sig_s - self.min_sig_s) * np.random.random_sample() + self.min_sig_s
            sample = cv2.bilateralFilter(sample, self.kernel_size_w, final_sig_c, final_sig_s)

        return sample


class TransformWhileSampling(object):

    def __init__(self, transform):
        self.transform = transform


    def __call__(self, sample):

        x1 = self.transform(sample)
        x2 = self.transform(sample)

        return x1, x2