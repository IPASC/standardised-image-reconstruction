"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Groehl
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.baseline_delay_and_sum import BaselineDelayAndSumAlgorithm


class TestDelayAndSum(TestClassBase):

    def test_delay_and_sum_reconstruction_is_running_through(self):
        self.run_tests(BaselineDelayAndSumAlgorithm(), **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": 1540
        })



