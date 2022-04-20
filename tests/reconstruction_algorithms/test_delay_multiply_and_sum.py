"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.delay_multiply_and_sum import DelayMultiplyAndSumAlgorithm


class TestDelayMultiplyAndSum(TestClassBase):

    # General parameters
    lowcut = None
    highcut = None
    order = 9
    envelope = False
    envelope_type = "hilbert"
    spacing_m = 0.0001
    speed_of_sound_m_s = 1540
    fnumber = 0
    signed_dmas = False

    def back_project(self, image_idx=0, visualise=True):
        return self.run_tests(DelayMultiplyAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "envelope": self.envelope,
            "fnumber": self.fnumber,
            "envelope_type": self.envelope_type,
            "signed_dmas": self.signed_dmas
        })
