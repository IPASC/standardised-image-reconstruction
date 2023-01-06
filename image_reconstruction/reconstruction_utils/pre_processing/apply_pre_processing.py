"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-License-Identifier: MIT
"""

from image_reconstruction.reconstruction_utils.pre_processing.bandpass_filter import butter_bandpass_filter
from image_reconstruction.reconstruction_utils.pre_processing.interpolation import interpolate_time_series
from image_reconstruction.reconstruction_utils.pre_processing.signal_scaling import scale_time_series


def apply_pre_processing(time_series_data, detection_elements, sampling_rate, **kwargs):
    """
    Applies all preprocessing steps as defined by the given keywod arguments.

    :param time_series_data: np.ndarray with the following definition [detectors, time samples]
    :param detection_elements: np.ndarray
    :param sampling_rate: float
    :param kwargs: dict
    :return: (time_series_data, detection_elements, sampling_rate)
        This function returns all parameters that can change during the preprocessing.
    """

    lowcut = None
    if "lowcut" in kwargs:
        lowcut = kwargs["lowcut"]

    highcut = None
    if "highcut" in kwargs:
        highcut = kwargs["highcut"]

    filter_order = 5
    if "filter_order" in kwargs:
        filter_order = kwargs["filter_order"]

    detector_interpolation_factor = 1
    if "detector_interpolation_factor" in kwargs:
        detector_interpolation_factor = kwargs["detector_interpolation_factor"]

    time_interpolation_factor = 1
    if "time_interpolation_factor" in kwargs:
        time_interpolation_factor = kwargs["time_interpolation_factor"]

    scaling_method = None
    if "scaling_method" in kwargs:
        scaling_method = kwargs["scaling_method"]

    if scaling_method is not None:
        time_series_data = scale_time_series(time_series_data, scaling_method)

    if lowcut is not None or highcut is not None:
        time_series_data = butter_bandpass_filter(signal=time_series_data,
                                                  sampling_rate=sampling_rate,
                                                  lowcut=lowcut,
                                                  highcut=highcut,
                                                  order=filter_order)

    if detector_interpolation_factor != 1 or time_interpolation_factor != 1:
        time_series_data, positions = interpolate_time_series(time_series_data=time_series_data,
                                                              detector_positions=detection_elements["positions"],
                                                              detector_interpolation_factor=detector_interpolation_factor,
                                                              time_interpolation_factor=time_interpolation_factor)
        sampling_rate = sampling_rate * time_interpolation_factor
        detection_elements["positions"] = positions

    return time_series_data, detection_elements, sampling_rate
