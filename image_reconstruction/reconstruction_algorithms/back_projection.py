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

    def get_name(self) -> str:
        return "Delay-And-Sum"

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       sampling_rate: float,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
        Implementation of a delay and sum algorithm.

        The baseline implementation reflects the reconstruction algorithm described by Xu and Wang, 2005:

            Xu, M., & Wang, L. V. (2005). Universal back-projection algorithm for photoacoustic computed 
            tomography. Physical Review E, 71(1), 016706.

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
            ** 'p_factor' the p-factor TODO include paper reference
            ** 'p_SCF' the SCF-factor as described in these papers::

                    Kalkhoran, Mohammad Azizian, et al.
                    "Volumetric pulse echo and optoacoustic imaging by elaborating a weighted synthetic aperture technique."
                    2015 IEEE International Ultrasonics Symposium (IUS). IEEE, 2015.

                    Camacho, Jorge, Montserrat Parrilla, and Carlos Fritsch.
                    "Phase coherence imaging."
                    IEEE transactions on ultrasonics, ferroelectrics, and frequency control 56.5 (2009): 958-974.

            ** 'p_PCF' the PCF-factor as described in these papers::

                    Kalkhoran, Mohammad Azizian, et al.
                    "Volumetric pulse echo and optoacoustic imaging by elaborating a weighted synthetic aperture technique."
                    2015 IEEE International Ultrasonics Symposium (IUS). IEEE, 2015.

                    Camacho, Jorge, Montserrat Parrilla, and Carlos Fritsch.
                    "Phase coherence imaging."
                    IEEE transactions on ultrasonics, ferroelectrics, and frequency control 56.5 (2009): 958-974.

            ** 'fnumber' the fnumber as described in this paper::

                    Perrot, Vincent, et al.
                    "So you think you can DAS?
                    A viewpoint on delay-and-sum beamforming." Ultrasonics 111 (2021): 106309.

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
                                        sampling_rate=sampling_rate,
                                        field_of_view=field_of_view,
                                        spacing_m=spacing_m,
                                        speed_of_sound_in_m_per_s=speed_of_sound_in_m_per_s,
                                        fnumber=fnumber,
                                        p_scf=p_scf,
                                        p_factor=p_factor,
                                        p_pcf=p_pcf)

        return reconstructed
