"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.FFT_based import FFTbasedAlgorithm


class TestFFTbasedJaeger(TestClassBase):

    speed_of_sound_m_s = 1540
    pitch_size = 0.000315 # 0.3mm
    time_delay = 0
    zero_padding_X = 1
    zero_padding_Y = 1
    coefficientT = 5
    image_sampling = 1


    def fftbasedJaeger(self, image_idx=0, visualise=True):
        return self.run_tests(FFTbasedAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "element_size": self.pitch_size,
            "delay": self.time_delay,
            "zeroX": self.zero_padding_X,
            "zeroT": self.zero_padding_Y,
            "coeffT": self.coefficientT,
            "samplingX": self.image_sampling
        })
