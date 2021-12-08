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
from image_reconstruction.reconstruction_algorithms.back_projection import BackProjection


class TestDelayAndSum(TestClassBase):

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
    fnumber = 2

    def test_vanilla_delay_and_sum_reconstruction_is_running_through(self, image_idx=0, visualise=True):
        return self.run_tests(BaselineDelayAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s
        })

    def test_delay_and_sum_reconstruction_bandpass_is_running_through(self, image_idx=0, visualise=True):
        return self.run_tests(BaselineDelayAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order
        })

    def test_delay_and_sum_reconstruction_bandpass_pre_envelope_is_running_through(self, image_idx=0, visualise=True):
        return self.run_tests(BaselineDelayAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "envelope_time_series": True
        })

    def test_delay_and_sum_reconstruction_bandpass_post_envelope_is_running_through(self, image_idx=0, visualise=True):
        return self.run_tests(BaselineDelayAndSumAlgorithm(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": 5000,
            "highcut": 7000000,
            "order": 9,
            "envelope_reconstructed": True
        })

    def test_delay_and_sum_reconstruction_is_running_through_fnumber(self, image_idx=0, visualise=True):
        return self.run_tests(BaselineDelayAndSumAlgorithmFnumber(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "envelope_time_series": self.envelope,
            "fnumber": self.fnumber
        })

    def test_delay_and_sum_reconstruction_is_running_through_pDAS(self, image_idx=0, visualise=True):
        return self.run_tests(BaselineDelayAndSumAlgorithmpDAS(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "envelope_time_series": self.envelope,
            "fnumber": self.fnumber
        })

    def test_delay_and_sum_reconstruction_is_running_through_SCF(self, image_idx=0, visualise=True):
        return self.run_tests(BaselineDelayAndSumAlgorithmSCF(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "envelope_time_series": self.envelope,
            "p_factor": self.p_factor,
            "p_SCF": self.p_SCF,
            "fnumber": self.fnumber
        })

    def back_project(self, image_idx=0, visualise=True):
        return self.run_tests(BackProjection(), image_idx=image_idx, visualise=visualise, **{
            "spacing_m": self.spacing_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
            "envelope": self.envelope,
            "p_factor": self.p_factor,
            "p_SCF": self.p_SCF,
            "fnumber": self.fnumber,
            "envelope_type": self.envelope_type
        })
