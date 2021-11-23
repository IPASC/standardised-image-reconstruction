"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Groehl
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.baseline_delay_and_sum import BaselineDelayAndSumAlgorithm
from image_reconstruction.reconstruction_algorithms.baseline_delay_and_sum_fnumber import BaselineDelayAndSumAlgorithmFnumber
from image_reconstruction.reconstruction_algorithms.baseline_delay_and_sum_pDAS import BaselineDelayAndSumAlgorithmpDAS
from image_reconstruction.reconstruction_algorithms.baseline_delay_and_sum_SCF import BaselineDelayAndSumAlgorithmSCF


class TestDelayAndSum(TestClassBase):

    p_factor = 1
    fnumber = 2

    def test_delay_and_sum_reconstruction_is_running_through(self):
        self.run_tests(BaselineDelayAndSumAlgorithm(), **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": 1540
        })

    def test_delay_and_sum_reconstruction_is_running_through_fnumber(self):
        self.run_tests(BaselineDelayAndSumAlgorithmFnumber(), **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": 1540,
            "fnumber": self.fnumber
        })

    def test_delay_and_sum_reconstruction_is_running_through_pDAS(self):
        self.run_tests(BaselineDelayAndSumAlgorithmpDAS(), **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": 1540,
            "p_factor": self.p_factor,
            "fnumber": self.fnumber
        })

    def test_delay_and_sum_reconstruction_is_running_through_SCF(self):
        self.run_tests(BaselineDelayAndSumAlgorithmSCF(), **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": 1540,
            "p_factor": self.p_factor,
            "fnumber": self.fnumber
        })
