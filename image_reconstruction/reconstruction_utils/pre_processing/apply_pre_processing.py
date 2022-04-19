

from image_reconstruction.reconstruction_utils.pre_processing.bandpass_filter import butter_bandpass_filter


def apply_pre_processing(time_series_data, sampling_rate, **kwargs):

    lowcut = None
    if "lowcut" in kwargs:
        lowcut = kwargs["lowcut"]

    highcut = None
    if "highcut" in kwargs:
        highcut = kwargs["highcut"]

    filter_order = 5
    if "filter_order" in kwargs:
        filter_order = kwargs["filter_order"]

    if lowcut is not None or highcut is not None:
        time_series_data = butter_bandpass_filter(signal=time_series_data,
                                                  sampling_rate=sampling_rate,
                                                  lowcut=lowcut,
                                                  highcut=highcut,
                                                  order=filter_order)


    return time_series_data
