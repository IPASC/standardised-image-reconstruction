"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-License-Identifier: MIT
"""

from image_reconstruction.reconstruction_algorithms import BackProjection, FftBasedJaeger2007, FFTbasedHauptmann2018


def reconstruct_ipasc_hdf5(path_to_hdf5: str, algorithms_settings_tuples: list):

    reconstructed_images = []

    for (algorithm, settings) in algorithms_settings_tuples:
        reconstructed_images.append(algorithm.reconstruct_time_series_data(path_to_ipasc_hdf5=path_to_hdf5, **settings))

    return reconstructed_images


if __name__ == "__main__":

    settings = {
            "spacing_m": 0.0001,
            "speed_of_sound_m_s": 1540,
            "lowcut": 5000,
            "highcut": 7e6,
            "order": 9,
            "envelope": False,
            "p_factor": 1,
            "p_SCF": 1,
            "p_PCF": 0,
            "fnumber": 1,
            "envelope_type": None,
            "delay": 0,
            "zeroX": 1,
            "zeroT": 1,
            "fourier_coefficients_dim": 5,
        }
    algorithms = [(BackProjection(), settings),
                  (FftBasedJaeger2007(), settings),
                  (FFTbasedHauptmann2018(), settings)]

    reconstruct_ipasc_hdf5("../1BdSLl4BSxpxXDwWcBKKVV4nHULPe7IS8_ipasc.hdf5", algorithms)