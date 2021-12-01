from scipy.signal import hilbert, hilbert2
import numpy as np


def hilbert_transform_1D(signal, axis=-1):
    """

    :param signal:
    :param axis: The axis the hilbert transform should be computed on
    :return:
    """
    return np.abs(hilbert(signal, axis=axis))


def hilbert_transform_2D(signal):
    """

    :param signal: a NxM numpy ndarray
    :return: the 2D hilbert transform of the signal
    """
    return np.abs(hilbert2(signal))
