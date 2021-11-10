"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)

Please note that the code here is an adapted version of the code
published in the SIMPA repository also under the MIT license:
https://github.com/CAMI-DKFZ/simpa

SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""
import numpy as np
import torch
from image_reconstruction.reconstruction_algorithms import ReconstructionAlgorithm


class BaselineDelayAndSumAlgorithmpDAS(ReconstructionAlgorithm):

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
        Implementation of a baseline delay and sum algorithm without any additional features.

        :param time_series_data: A 2D numpy array with the following internal array definition:
                                [detectors, time samples]
        :param detection_elements: A dictionary that describes the detection geometry.
                                   The dictionary contains three entries:
                                   ** "positions": The positions of the detection elements relative to the field of view
                                   ** "orientations": The orientations of the detection elements
                                   ** "sizes": The sizes of the detection elements.
        :param field_of_view: A 1D 6 element-long numpy array that contains the extent of the field of view in x, y and
                              z direction in the same coordinate system as the detection element positions.
        :param kwargs: the list of parameters for the delay and sum reconstruction includes the following parameters:
            ** 'spacing_m' the target isotropic reconstruction spacing in units of meters
            ** 'speed_of_sound_m_s' the target speed of sound in units of meters per second
        :return:
        """

        # parse kwargs with sensible defaults
        speed_of_sound_in_m_per_s = 1540
        if "speed_of_sound_m_s" in kwargs:
            speed_of_sound_in_m_per_s = kwargs["speed_of_sound_m_s"]

        spacing_m = 0.0005
        if "spacing_m" in kwargs:
            spacing_m = kwargs["spacing_m"]

        torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        time_spacing_in_ms = self.ipasc_data.get_sampling_rate()
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

        values, _ = self.compute_delay_and_sum_values(time_series_data,
                                                      sensor_positions,
                                                      field_of_view_voxels,
                                                      spacing_m,
                                                      speed_of_sound_in_m_per_s,
                                                      time_spacing_in_ms,
                                                      torch_device)

        p = 2.5 # the p power of the DAS
        # We do sign(s)*abs(s)^(1/p)
        values = torch.mul( torch.sign(values), torch.pow(torch.abs(values), 1/p))

        # we do the sum
        _sum = torch.sum(values, dim=3)

        # we come back in the correct domain : sign(s)*abs(s)^(p)
        _sum = torch.mul( torch.sign(_sum), torch.pow(torch.abs(_sum), p))
        counter = torch.count_nonzero(values, dim=3)

        torch.divide(_sum, counter, out=output)

        reconstructed = output.cpu().numpy()

        return reconstructed

    def compute_delay_and_sum_values(self,
                                     time_series_data: torch.tensor,
                                     sensor_positions: torch.tensor,
                                     field_of_view_voxels: np.ndarray,
                                     spacing_in_m: float,
                                     speed_of_sound_in_m_per_s: float,
                                     time_spacing_in_s: float,
                                     torch_device: torch.device) -> (torch.tensor, int):
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

        :return: returns a tuple with
                 ** values (torch tensor) of the time series data corrected for delay and sensor positioning, ready to be summed up
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
                            (zz * spacing_in_m - sensor_positions[:, 1][jj]) ** 2) \
                 / (speed_of_sound_in_m_per_s * time_spacing_in_s)


        # perform index validation
        invalid_indices = torch.where(torch.logical_or(delays < 0, delays >= float(time_series_data.shape[1])))
        torch.clip_(delays, min=0, max=time_series_data.shape[1] - 1)

        delays = (torch.round(delays)).long()
        values = time_series_data[jj, delays]

        # Add fNumber
        fnumber = 0 # The value must be changed in other places...
        if (fnumber>0):
            invalid_indices2 = torch.where(torch.logical_not(torch.abs(xx * spacing_in_m - sensor_positions[:, 0][jj]) \
                < (zz * spacing_in_m - sensor_positions[:, 1][jj]) /fnumber/2))
        else:
            invalid_indices2 = invalid_indices


        # set values of invalid indices to 0 so that they don't influence the result
        values[invalid_indices] = 0
        values[invalid_indices2] = 0

        del delays  # free memory of delays

        return values, n_sensor_elements
