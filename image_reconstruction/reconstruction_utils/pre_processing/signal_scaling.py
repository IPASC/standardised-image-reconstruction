"""
SPDX-FileCopyrightText: 2023 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2023 Janek Gröhl
SPDX-License-Identifier: MIT
"""

import numpy as np


def scale_time_series(time_series_data, method=None):
    """

    :param time_series_data: np.ndarray
    :param method: str, int
        The method for scaling the time series. The main purpose of this is to remove the noise floor
        from each individual detection element to have it center around zero. In literature, usually
        subtraction of the mean is used, but this function also supports the median or any percentile if
        this argument is given a number between 0 and 100.
        The supported arguments are:

        - "mean": default
        - "median"
        - int: (0-100) will subtract a specific percentile

    :return:
    """

    if method is None or method == "mean":
        return time_series_data - np.mean(time_series_data, axis=1, keepdims=True)
    elif method == "median":
        return time_series_data - np.median(time_series_data, axis=1, keepdims=True)
    elif isinstance(method, (float, int)):
        return time_series_data - np.percentile(time_series_data, method, axis=1, keepdims=True)
    else:
        raise AssertionError(f"Illegal argument for method: {method} of type {type(method)}")
