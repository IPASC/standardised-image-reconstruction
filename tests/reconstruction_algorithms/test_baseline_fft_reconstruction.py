"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Jenni Poimala
SPDX-FileCopyrightText: 2021 Andreas Hauptmann
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.fft_based import FFTbasedHauptmann2018


class TestFFTbasedHauptmann(TestClassBase):

    # General parameters
    lowcut = 5000
    highcut = 7e6
    order = 9
    envelope = False
    envelope_type = None
    spacing_m = 0.0001
    speed_of_sound_m_s = 1540
  

    def fft_recon(self, image_idx=0, visualise=True):
        return self.run_tests(FFTbasedHauptmann2018(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "envelope": self.envelope,
            "envelope_type": self.envelope_type
        })
