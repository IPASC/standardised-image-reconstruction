import numpy as np
from scipy.interpolate import interp2d


def interpolate_time_series(time_series_data, detector_positions,
                            detector_interpolation_factor: int = 1,
                            time_interpolation_factor: float = 1):

    new_detector_positions = np.asarray([detector_positions[i, :] for i in range(len(detector_positions)) for _ in
                                         range(detector_interpolation_factor)])
    new_time_series_data = np.asarray([time_series_data[i, :] for i in range(len(time_series_data)) for _ in
                                         range(detector_interpolation_factor)])

    x = np.arange(0, len(new_detector_positions), 1)
    y = np.arange(0, len(time_series_data[0, :]), 1)

    f = interp2d(y, x, new_time_series_data, kind="linear")

    new_x = np.arange(0, len(new_detector_positions), 1)
    new_y = np.arange(0, len(time_series_data[0, :]), 1/time_interpolation_factor)

    return f(new_y, new_x), new_detector_positions
