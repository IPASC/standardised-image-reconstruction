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
from image_reconstruction.reconstruction_utils.beamforming import compute_delay_and_sum_values


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
