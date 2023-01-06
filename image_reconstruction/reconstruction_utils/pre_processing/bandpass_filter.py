"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-License-Identifier: MIT
"""

from scipy.signal import butter, lfilter


def _butter_bandpass(sampling_rate, lowcut=None, highcut=None, order=5):
    """
    Creates the b, a filter definitions for the butter bandpass filter.

    Parameters
    ----------
    sampling_rate : int
    lowcut : int
    highcut : int
    order : int

    Returns
    --------
    the b, a filter definitions
    """
    nyq = 0.5 * sampling_rate
    if lowcut is None:
        low = 0.000001
    else:
        low = lowcut / nyq
    if highcut is None:
        high = 0.999999999
    else:
        high = highcut / nyq
        if high > 1 - 1e-10:
            high = 1 - 1e-10
    b, a = butter(N=order, Wn=[low, high], btype='band')
    return b, a


def butter_bandpass_filter(signal, sampling_rate, lowcut=None, highcut=None, order=5):
    """
    Applies a butter bandpass filter of the specified order around the specified
    lowcut (highpass) and highcut (lowpass) frequencies.

    Parameters
    ----------
    signal: np.ndarray
    sampling_rate: int
    lowcut: int
    highcut: int
    order: int

    Returns
    -------
    The signal after bandpass filtering using a butter bandpass filter
    """
    b, a = _butter_bandpass(sampling_rate=sampling_rate,
                            lowcut=lowcut,
                            highcut=highcut, order=order)
    y = lfilter(b, a, signal)
    return y
