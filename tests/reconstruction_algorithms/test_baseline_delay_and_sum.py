"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Groehl
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.baseline_delay_and_sum import BaselineDelayAndSumAlgorithm


class TestDelayAndSum(TestClassBase):

    p_factor = 1
    fnumber = 2

    def test_vanilla_delay_and_sum_reconstruction_is_running_through(self, image_idx=0, visualise=True,
                                                                     speed_of_sound=1540):
        return self.run_tests(BaselineDelayAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": speed_of_sound
        })

    def test_delay_and_sum_reconstruction_bandpass_is_running_through(self, image_idx=0, visualise=True,
                                                                     speed_of_sound=1540):
        return self.run_tests(BaselineDelayAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": speed_of_sound,
            "lowcut": 5000,
            "highcut": 7000000,
            "order": 9
        })

    def test_delay_and_sum_reconstruction_bandpass_pre_envelope_is_running_through(self, image_idx=0, visualise=True,
                                                                     speed_of_sound=1540):
        return self.run_tests(BaselineDelayAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": speed_of_sound,
            "lowcut": 5000,
            "highcut": 7000000,
            "order": 9,
            "envelope_time_series": True
        })

    def test_delay_and_sum_reconstruction_bandpass_post_envelope_is_running_through(self, image_idx=0, visualise=True,
                                                                     speed_of_sound=1540):
        return self.run_tests(BaselineDelayAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": speed_of_sound,
            "lowcut": 5000,
            "highcut": 7000000,
            "order": 9,
            "envelope_reconstructed": True
        })
