import os
import cv2
import numpy
from PIL import Image


def hist_open(file_path):
    '''
    Opens the file at file_path and returns its contents if it's a histogram.  Otherwise, assumes it contains an image and returns its histogram.
    '''
    img = Image.open(file_path)
    img = numpy.asarray(img)
    img = img.transpose((2, 0, 1))
    histB = numpy.histogram(img[0], bins=2 ** 8, range=(0, 2 ** 8))[0]
    histG = numpy.histogram(img[1], bins=2 ** 8, range=(0, 2 ** 8))[0]
    histR = numpy.histogram(img[2], bins=2 ** 8, range=(0, 2 ** 8))[0]
    RGB = numpy.stack([histR, histG, histB])
    return RGB


def hist_normalize(histogram):
    '''
    Normalizes a histogram, putting its bins in the range [0, 1].  The sum of all bins is 1.
    '''
    _ = histogram[0] / numpy.sum(histogram[0], axis=0)
    __ = histogram[1] / numpy.sum(histogram[1], axis=0)
    ___ = histogram[2] / numpy.sum(histogram[2], axis=0)
    hist = numpy.stack([_, __, ___])
    return hist


def hist_cdf(histogram):
    '''
    Computes the cumulative distribution function of a histogram.
    '''

    return numpy.cumsum(histogram) / numpy.sum(histogram)


def hist_resize(histogram, bins):
    '''
    Resizes a histogram to have a new number of bins.
    '''

    assert bins > 0
    if len(histogram) < bins:
        bin_map = numpy.floor(numpy.arange(0, bins, bins / len(histogram))).astype(numpy.int_)
        bin_widths = numpy.append(bin_map[1:], bins) - bin_map
        return numpy.repeat(histogram / bin_widths, bin_widths)
    # Interpolation might look nicer, but the interpolation points would need to be less than histogram / bin_widths (otherwise extra mass is added as a triangle formed between interpolation points).
    # return numpy.interp(numpy.arange(bins), bin_map, histogram / bin_widths)
    elif len(histogram) > bins:
        return numpy.add.reduceat(histogram, numpy.floor(numpy.arange(0, len(histogram), len(histogram) / bins)).astype(
            numpy.int_))
    else:
        return histogram


def hist_save(histogram, save_path):
    '''
    Saves a histogram to a file at save_path.  Creates directories along the path as necessary.
    '''

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    numpy.save(save_path, histogram, allow_pickle=False)
