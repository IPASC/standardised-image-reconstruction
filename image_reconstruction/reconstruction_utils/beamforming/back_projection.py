"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT

Please note that the code here is based on the code
published in the SIMPA repository also under the MIT license:
https://github.com/CAMI-DKFZ/simpa
"""

import torch
import numpy as np
from scipy.signal import hilbert

def back_projection(time_series_data, detection_elements, sampling_rate,
                    field_of_view, spacing_m, speed_of_sound_in_m_per_s,
                    fnumber, p_scf, p_factor, p_pcf):
    """

    Parameters
    ----------
    time_series_data
    detection_elements
    sampling_rate
    field_of_view
    spacing_m
    speed_of_sound_in_m_per_s
    fnumber
    p_scf
    p_factor
    p_pcf

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

    values, _ = compute_delay_and_sum_values(time_series_data,
                                             sensor_positions,
                                             field_of_view_voxels,
                                             spacing_m,
                                             speed_of_sound_in_m_per_s,
                                             time_spacing_in_s,
                                             torch_device,
                                             fnumber)

    if p_pcf:
        # compute the phase -> we impose a fnumber = 0 for maximum phase calculation
        raw_phase = torch.from_numpy(hilbert(time_series_data.cpu().numpy(), axis=0)).to(torch_device)
        raw_phase = raw_phase.angle()
        # do beamforming with the phase -> we impose a fnumber = 0 for maximum phase calculation
        phase, _ = compute_delay_and_sum_values(raw_phase,
                                                sensor_positions,
                                                field_of_view_voxels,
                                                spacing_m,
                                                speed_of_sound_in_m_per_s,
                                                time_spacing_in_s,
                                                torch_device, 0)
        sigma = torch.std(phase, dim=3)
        sigma_A = torch.std(torch.add(phase, torch.mul(torch.sign(phase), torch.tensor(-np.pi))), dim=3)
        sf = torch.minimum(sigma, sigma_A)
        sigma_0 = np.pi / np.sqrt(3)
        _PCF = 1 - p_pcf / sigma_0 * sf
        _PCF[torch.where(_PCF < 0)] = 0  # equivalent to _PCF = max(0, 1-p_pcf/sigma_0*sf)

    if p_scf:
        # We extract and sum the sign of the value
        _SCF = torch.mean(torch.sign(values), dim=3)
        _SCF = torch.pow(torch.abs(1 - torch.sqrt(1 - torch.pow(_SCF, 2))), p_scf)

    # We do sign(s)*abs(s)^(1/p)
    values = torch.mul(torch.sign(values), torch.pow(torch.abs(values), 1 / p_factor))

    # we do the sum
    _sum = torch.sum(values, dim=3)

    # we come back in the correct domain : sign(s)*abs(s)^(p)
    _sum = torch.mul(torch.sign(_sum), torch.pow(torch.abs(_sum), p_factor))
    counter = torch.count_nonzero(values, dim=3)

    # We multiply with the SCF coeeficient
    if p_scf:
        _sum = torch.mul(_sum, _SCF)

    # We multiply with the PCF factor is necessary
    if p_pcf:
        _sum = torch.mul(_sum, _PCF)

    torch.divide(_sum, counter, out=output)

    return output.cpu().numpy()


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
                                    torch.arange(n_sensor_elements, device=torch_device))

    delays = torch.sqrt((yy * spacing_in_m - sensor_positions[:, 2][jj]) ** 2 +
                        (xx * spacing_in_m - sensor_positions[:, 0][jj]) ** 2 +
                        (zz * spacing_in_m - sensor_positions[:, 1][jj]) ** 2) / (speed_of_sound_in_m_per_s *
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
