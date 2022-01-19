"""
SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Gröhl
SPDX-FileCopyrightText: 2022 Mengjie Shi
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.FFT_based import FFTbasedAlgorithm


class TestFFTbasedJaeger(TestClassBase):

    speed_of_sound_m_s = 1540
    time_delay = 0
    zero_padding_transducer_dimension = 1
    zero_padding_time_dimension = 1
    coefficientT = 5
    image_sampling = 1

    def fftbasedJaeger(self, image_idx=0, visualise=True):
        return self.run_tests(FFTbasedAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "delay": self.time_delay,
            "zeroX": self.zero_padding_transducer_dimension,
            "zeroT": self.zero_padding_time_dimension,
            "coeffT": self.coefficientT,
            "samplingX": self.image_sampling
        })
