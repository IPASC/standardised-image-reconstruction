"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""
import numpy as np
from image_reconstruction.reconstruction_utils.beamforming import back_projection
from image_reconstruction.reconstruction_algorithms import ReconstructionAlgorithm


class BackProjection(ReconstructionAlgorithm):

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
        Implementation of a delay and sum algorithm.

        The baseline implementation reflects the reconstruction algorithm described by Xu and Wang, 2005::

            Xu, Minghua, and Lihong V. Wang.
            "Universal back-projection algorithm for photoacoustic computed tomography."
            Physical Review E 71.1 (2005): 016706.

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
            ** 'p_factor' the p-factor TODO include paper reference
            ** 'p_SCF' the SCF-factor TODO include paper reference
            ** 'p_PCF' the PCF-factor TODO include paper reference
            ** 'fnumber' the fnumber TODO include paper reference

        Returns
        -------
        A reconstructed image
        """

        time_series_data = time_series_data.astype(float)

        # parse kwargs with sensible defaults
        speed_of_sound_in_m_per_s = 1540
        if "speed_of_sound_m_s" in kwargs:
            speed_of_sound_in_m_per_s = kwargs["speed_of_sound_m_s"]
        elif self.ipasc_data.get_speed_of_sound() is not None:
            speed_of_sound_in_m_per_s = self.ipasc_data.get_speed_of_sound()

        spacing_m = 0.0005
        if "spacing_m" in kwargs:
            spacing_m = kwargs["spacing_m"]

        p_factor = 1
        if "p_factor" in kwargs:
            p_factor = kwargs["p_factor"]

        p_scf = 0
        if "p_SCF" in kwargs:
            p_scf = kwargs["p_SCF"]

        p_pcf = 0
        if "p_PCF" in kwargs:
            p_pcf = kwargs["p_PCF"]

        fnumber = 0
        if "fnumber" in kwargs:
            fnumber = kwargs["fnumber"]

        reconstructed = back_projection(time_series_data=time_series_data,
                                        detection_elements=detection_elements,
                                        sampling_rate=self.ipasc_data.get_sampling_rate(),
                                        field_of_view=field_of_view,
                                        spacing_m=spacing_m,
                                        speed_of_sound_in_m_per_s=speed_of_sound_in_m_per_s,
                                        fnumber=fnumber,
                                        p_scf=p_scf,
                                        p_factor=p_factor,
                                        p_pcf=p_pcf)

        return reconstructed
