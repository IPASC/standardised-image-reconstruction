"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-License-Identifier: MIT

Please note that the code here is based on the code
published in the SIMPA repository also under the MIT license:
https://github.com/IMSY-DKFZ/simpa
"""
import torch


def get_apodisation_factor(apodization_method: str = "box",
                           dimensions: tuple = None, n_sensor_elements=None,
                           device: torch.device = 'cpu') -> torch.tensor:
    """
    Construct apodization factors according to `apodization_method` [hann, hamming or box apodization (default)]
    for given dimensions and `n_sensor_elements`.

    :param apodization_method: (str) Apodization method, one of hann, hamming and box (default)
    :param dimensions: (tuple) size of each dimension of reconstructed image as int, might have 2 or 3 entries.
    :param n_sensor_elements: (int) number of sensor elements
    :param device: (torch device) PyTorch tensor device
    :return: (torch tensor) tensor with apodization factors which can be multipied with DAS values
    """

    if dimensions is None or n_sensor_elements is None:
        raise AttributeError("dimensions and n_sensor_elements must be specified and not be None")

    # hann window
    if apodization_method == "hann":
        hann = torch.hann_window(n_sensor_elements, device=device)
        output = hann.expand(dimensions + (n_sensor_elements,))
    # hamming window
    elif apodization_method == "hamming":
        hamming = torch.hamming_window(n_sensor_elements, device=device)
        output = hamming.expand(dimensions + (n_sensor_elements,))
    # box window apodization as default
    else:
        output = torch.ones(dimensions + (n_sensor_elements,), device=device)

    return output