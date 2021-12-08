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

def log_compression(signal, axis=-1, dynamic=40):
    """

    :param signal:
    :param axis: The axis the hilbert transform should be computed on
    :param dynamic: the dynmaic in dB to be displayed
    :return:
    """
    # do the hilbert transform
    env = hilbert_transform_1D(signal, axis)
    # do 20log10 on the normalized image
    env = 20*np.log10(env/np.nanmax(env))
    # put to zeros everythin that is outside the desired dynamic
    env[np.where(env<-dynamic)] = -dynamic

    return env
