"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from scipy.signal import hilbert, hilbert2
import numpy as np


def hilbert_transform_1_d(signal, axis=-1):
    """

    Parameters
    ----------
    signal: np.ndarray a NxM numpy ndarray
    axis: int The axis the hilbert transform should be computed on

    Returns
    -------
    The 1D hilbert transform along a specified axis
    """
    return np.abs(hilbert(signal, axis=axis))


def hilbert_transform_2_d(signal):
    """

    Parameters
    ----------
    signal: np.ndarray a NxM numpy ndarray

    Returns
    -------
    the 2D hilbert transform of the signal
    """
    return np.abs(hilbert2(signal))


def zero_forcing(signal, threshold=1e-20):
    """

    Parameters
    ----------
    signal: np.ndarray a NxM numpy ndarray
    threshold: float the cutoff value for the zero forcing (default: 1e-20)

    Returns
    -------
    the signal, where no value is smaller than threshold
    """
    signal[signal < threshold] = threshold
    return signal


def absolute_value(signal):
    """

    Parameters
    ----------
    signal np.ndarray a NxM numpy ndarray

    Returns
    -------
    the absolute values of the input signal
    """
    return np.abs(signal)
