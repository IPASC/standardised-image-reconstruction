"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-License-Identifier: MIT
"""

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, sampling_rate, order=5):
    """

    Parameters
    ----------
    lowcut : int
    highcut : int
    sampling_rate : int
    order : int

    Returns
    --------
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

    b, a = butter(N=order, Wn=[low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, sampling_rate, order=order)
    y = lfilter(b, a, data)
    return y
