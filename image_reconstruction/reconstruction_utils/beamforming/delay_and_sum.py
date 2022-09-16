"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT

Please note that the code here is based on the code
published in the SIMPA repository also under the MIT license:
https://github.com/IMSY-DKFZ/simpa
"""

import torch
import numpy as np


def compute_delay_and_sum_values(time_series_data: torch.tensor,
                                 sensor_positions: torch.tensor,
                                 field_of_view_voxels: np.ndarray,
                                 spacing_in_m: float,
                                 speed_of_sound_in_m_per_s: float,
                                 time_spacing_in_s: float,
                                 torch_device: torch.device,
                                 fnumber: float = 1.0) -> (torch.tensor, int):
    """
    Perform the core computation of Delay and Sum, without summing up the delay dependend values.

    :param time_series_data: A 2D numpy array with the following internal array definition:
                            [detectors, time samples]
    :param sensor_positions: A numpy array with the positions of all
    :param field_of_view_voxels: A numpy array containing the field of view in voxels
    :param spacing_in_m: Target spacing in units of meters
    :param speed_of_sound_in_m_per_s: Speed of sound in units of meters per second
    :param time_spacing_in_s: Inverse sampling rate in units of seconds
    :param torch_device: the pytorch device to compute everything on
    :param fnumber: the fnumber parameter to limit the sum angles

    :return: returns a tuple with
             ** values (torch tensor) of the time series data corrected for delay and sensor positioning, ready to
             be summed up
             ** n_sensor_elements (int) which might be used for later computations
    """

    n_sensor_elements = time_series_data.shape[0]

    xx, yy, zz, jj = torch.meshgrid(torch.arange(field_of_view_voxels[0],
                                                 field_of_view_voxels[1], device=torch_device)
                                    if (field_of_view_voxels[1] - field_of_view_voxels[0])
                                    >= 1 else torch.arange(1, device=torch_device),
                                    torch.arange(field_of_view_voxels[2],
                                                 field_of_view_voxels[3], device=torch_device)
                                    if (field_of_view_voxels[3] - field_of_view_voxels[2])
                                    >= 1 else torch.arange(1, device=torch_device),
                                    torch.arange(field_of_view_voxels[4],
                                                 field_of_view_voxels[5], device=torch_device)
                                    if (field_of_view_voxels[5] - field_of_view_voxels[4])
                                    >= 1 else torch.arange(1, device=torch_device),
                                    torch.arange(n_sensor_elements, device=torch_device),
                                    indexing="ij")

    delays = torch.sqrt((yy * spacing_in_m - sensor_positions[:, 1][jj]) ** 2 +
                        (xx * spacing_in_m - sensor_positions[:, 0][jj]) ** 2 +
                        (zz * spacing_in_m - sensor_positions[:, 2][jj]) ** 2) / (speed_of_sound_in_m_per_s *
                                                                                  time_spacing_in_s)

    # perform index validation
    invalid_indices = torch.where(torch.logical_or(delays < 0, delays >= float(time_series_data.shape[1])))
    torch.clip_(delays, min=0, max=time_series_data.shape[1] - 1)

    delays = (torch.round(delays)).long()
    values = time_series_data[jj, delays]
    values[invalid_indices] = 0

    # Add fNumber
    if fnumber > 0:
        values[torch.where(torch.logical_not(torch.abs(xx * spacing_in_m - sensor_positions[:, 0][jj])
               < (zz * spacing_in_m - sensor_positions[:, 1][jj]) / fnumber / 2))] = 0

    del delays  # free memory of delays

    return values, n_sensor_elements
