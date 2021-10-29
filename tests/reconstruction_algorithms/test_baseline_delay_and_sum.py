"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.baseline_delay_and_sum import BaselineDelayAndSumAlgorithm
from image_reconstruction.baseline_delay_and_sum_fnumber import BaselineDelayAndSumAlgorithmFnumber


class TestDelayAndSum(TestClassBase):

    def test_delay_and_sum_reconstruction_is_running_through(self):
        reco = BaselineDelayAndSumAlgorithm()
        result = reco.reconstruct_time_series_data(self.ipasc_hdf5_file_path, **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": 1540
        })
        self.visualise_result(result)

    def test_delay_and_sum_reconstruction_is_running_through_fnumber(self):
        reco = BaselineDelayAndSumAlgorithmFnumber()
        result = reco.reconstruct_time_series_data(self.ipasc_hdf5_file_path, **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": 1540
        })
        self.visualise_result(result)
