"""
SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Gröhl
SPDX-FileCopyrightText: 2022 Mengjie Shi
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.fft_based import FftBasedJaeger2007


class TestFFTbasedJaeger(TestClassBase):

    lowcut = 5000
    highcut = 7e6
    order = 9
    speed_of_sound_m_s = 1540
    time_delay = 0
    zero_padding_transducer_dimension = 1
    zero_padding_time_dimension = 1
    coefficientT = 5
    envelope = False
    envelope_type = None
    spacing_m = 0.0001

    def fftbasedJaeger(self, image_idx=0, visualise=True):
        return self.run_tests(FftBasedJaeger2007(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "delay": self.time_delay,
            "zeroX": self.zero_padding_transducer_dimension,
            "zeroT": self.zero_padding_time_dimension,
            "fourier_coefficients_dim": self.coefficientT,
            "envelope": self.envelope,
            "envelope_type": self.envelope_type,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order
        })
