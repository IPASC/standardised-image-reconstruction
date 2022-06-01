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
from image_reconstruction.reconstruction_utils.beamforming.delay_and_sum import compute_delay_and_sum_values


def delay_multiply_and_sum(time_series_data, detection_elements, sampling_rate,
                           field_of_view, spacing_m, speed_of_sound_in_m_per_s,
                           fnumber, signed_dmas):
    """

    Parameters
    ----------
    time_series_data: numpy.ndarray
        The time series data for each detection element
    detection_elements: dict
        A dictionary containing metadata on the detection elements. Must at least contain a numpy array with the detector "positions".
    sampling_rate: float
        The sampling rate of the photoacoustic device in Hz
    field_of_view: numpy.ndarray
        The definition of the field of view the image should be reconstructed in. Defined as: [xmin, xmax, ymin, ymax, zmin, zmax].
    spacing_m: float
        The target voxel spacing in the reconstructed image
    speed_of_sound_in_m_per_s: float
        The assumed average speed of sound in the medium
    fnumber: float
        fnumber as defined in Perrot et al. 2021
    signed_dmas: bool
        whether to retain the sign of the corresponding DAS reconstruction as defined by Kirchner et al. 2018
       
    Returns
    -------

    """

    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    time_spacing_in_s = 1.0 / sampling_rate
    time_series_data = torch.from_numpy(time_series_data).to(torch_device)
    positions = detection_elements["positions"]
    sensor_positions = torch.from_numpy(positions).to(torch_device)

    field_of_view_voxels = np.round(field_of_view / spacing_m).astype(int)

    x_dim = (field_of_view_voxels[1] - field_of_view_voxels[0])
    y_dim = (field_of_view_voxels[3] - field_of_view_voxels[2])
    z_dim = (field_of_view_voxels[5] - field_of_view_voxels[4])

    # Just to make sure not to allocate a 0-dimensional array
    if x_dim < 1:
        x_dim = 1
    if y_dim < 1:
        y_dim = 1
    if z_dim < 1:
        z_dim = 1

    # construct output image
    output = torch.zeros((x_dim, y_dim, z_dim), dtype=torch.float32, device=torch_device)

    values, n_sensor_elements = compute_delay_and_sum_values(time_series_data,
                                                             sensor_positions,
                                                             field_of_view_voxels,
                                                             spacing_m,
                                                             speed_of_sound_in_m_per_s,
                                                             time_spacing_in_s,
                                                             torch_device,
                                                             fnumber)

    for x in range(x_dim):
        yy, zz, nn, mm = torch.meshgrid(torch.arange(y_dim, device=torch_device),
                                        torch.arange(z_dim, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device),
                                        torch.arange(n_sensor_elements, device=torch_device),
                                        indexing="ij")
        multiplied_signals = values[x, yy, zz, nn] * values[x, yy, zz, mm]
        multiplied_signals = torch.sign(multiplied_signals) * torch.sqrt(torch.abs(multiplied_signals))
        # only take upper triangle without diagonal and sum up along n and m axis (last two)
        output[x] = torch.triu(multiplied_signals, diagonal=1).sum(dim=(-1, -2))

    if signed_dmas:
        output *= torch.sign(torch.sum(values, dim=3))

    return output.cpu().numpy()
