"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms import TestClassBase
from image_reconstruction.reconstruction_algorithms.time_reversal import TimeReversal


class TestTimeReversal(TestClassBase):

    # General parameters
    lowcut = 5000
    highcut = 7e6
    order = 9
    envelope = False
    envelope_type = None
    spacing_m = 0.0001
    speed_of_sound_m_s = 1540
    p_factor = 1
    p_SCF = 1
    p_PCF = 0
    fnumber = 2

    def back_project(self, image_idx=0, visualise=True):
        return self.run_tests(TimeReversal(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "envelope": self.envelope,
            "envelope_type": self.envelope_type
        })

if __name__ == "__main__":
    result1 = TestTimeReversal().back_project(0, visualise=True)