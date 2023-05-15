"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.back_projection import BackProjection
from image_reconstruction.reconstruction_algorithms.baseline_delay_and_sum_fnumber import BaselineDelayAndSumAlgorithmFnumber


class TestDelayAndSum(TestClassBase):

    # General parameters
    lowcut = 5000
    highcut = 7e6
    order = 9
    non_negativity_method = None
    spacing_m = 0.0001
    speed_of_sound_m_s = 1540
    p_factor = 1
    p_SCF = 1
    p_PCF = 0
    fnumber = 2
    scaling = "mean"

    def back_project(self, image_idx=0, visualise=True):
        return self.run_tests(BackProjection(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "p_factor": self.p_factor,
            "p_SCF": self.p_SCF,
            "p_PCF": self.p_PCF,
            "fnumber": self.fnumber,
            "non_negativity_method": self.non_negativity_method,
            "scaling_method": self.scaling
        })
