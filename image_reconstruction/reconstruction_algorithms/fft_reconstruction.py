"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Jenni Poimala
SPDX-FileCopyrightText: 2021 Andreas Hauptmann
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""
import numpy as np
from image_reconstruction.reconstruction_utils.fft import fft_reconstruction
from image_reconstruction.reconstruction_algorithms import ReconstructionAlgorithm


class FFTReconstruction(ReconstructionAlgorithm):

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
        Implementation of a baseline FFT based reconstruction algorithm without any additional features.

        Parameters
        ----------
        time_series_data: A 2D numpy array with the following internal array definition:
                          [detectors, time samples]
        detection_elements: A dictionary that describes the detection geometry.
                            The dictionary contains three entries:
                            ** "positions": The positions of the detection elements relative to the field of view
                            ** "orientations": The orientations of the detection elements
                            ** "sizes": The sizes of the detection elements.
        field_of_view: A 1D 6 element-long numpy array that contains the extent of the field of view in x, y and
                       z direction in the same coordinate system as the detection element positions.
        kwargs: the list of parameters for the delay and sum reconstruction includes the following parameters:
            ** 'spacing_m' the target isotropic reconstruction spacing in units of meters
            ** 'speed_of_sound_m_s' the target speed of sound in units of meters per second
            ** 'lowcut' the highpass frequency for the bandpass filter
            ** 'highcut' the lowpass frequency for the bandpass filter
            ** 'filter_order' the order of the butter filter
            ** 'envelope_type' the type of envelope detection to be performed

        Returns
        -------
        A reconstructed image
        """

        time_series_data = time_series_data.astype(float)

        # parse kwargs with sensible defaults
        speed_of_sound_in_m_per_s = 1540
        if "speed_of_sound_m_s" in kwargs:
            speed_of_sound_in_m_per_s = kwargs["speed_of_sound_m_s"]

        spacing_m = 0.0005
        if "spacing_m" in kwargs:
            spacing_m = kwargs["spacing_m"]


        reconstructed = fft_reconstruction(time_series_data, detection_elements, self.ipasc_data.get_sampling_rate(),
                                        field_of_view, spacing_m, speed_of_sound_in_m_per_s)

     

        return reconstructed
