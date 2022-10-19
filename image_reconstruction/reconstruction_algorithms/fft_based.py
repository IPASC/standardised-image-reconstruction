"""
SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Andreas Hauptmann
SPDX-FileCopyrightText: 2022 Jenni Poimala
SPDX-FileCopyrightText: 2022 Mengjie Shi
SPDX-FileCopyrightText: 2022 Janek Gröhl
SPDX-License-Identifier: MIT
"""

import numpy as np
from image_reconstruction.reconstruction_utils.beamforming.fft_based_jaeger_2007 import fft_based_jaeger_2d
from image_reconstruction.reconstruction_utils.beamforming.fft_based_hauptmann_2018 import fft_hauptmann_2d
from image_reconstruction.reconstruction_algorithms import ReconstructionAlgorithm


class FftBasedJaeger2007(ReconstructionAlgorithm):

    def get_name(self) -> str:
        return "FFT-based reconstruction (Jaeger 2007)"

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       sampling_rate: float,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
         Implementation of an FFT-based algorithm.

         The implementation reflects the reconstruction algorithm described by Jaeger et al., 2007::

            Jaeger, M., Schüpbach, S., Gertsch, A., Kitz, M., & Frenz, M. (2007). Fourier reconstruction 
            in optoacoustic imaging using truncated regularized inverse k-space interpolation. Inverse Problems, 
            23(6), S51.
          
         Parameters
         ----------

         :param time_series_data: A 2D numpy array with the following internal array definition:
                                 [detectors, time samples]
         :param detection_elements: A dictionary that describes the detection geometry.
                                    The dictionary contains three entries:
                                    ** "positions": The positions of the detection elements relative to the field of view
                                    ** "orientations": The orientations of the detection elements
                                    ** "sizes": The sizes of the detection elements.
         :param field_of_view: A 1D 6 element-long numpy array that contains the extent of the field of view in x, y and
                               z direction in the same coordinate system as the detection element positions.


         :param kwargs: the list of parameters for the fourier domain reconstruction includes the following parameters:
             ** 'speed_of_sound_m_s' the target speed of sound in units of meters per second
             ** 'delay': time delay from laser irradiation to signal acquisition start (default 0)
             ** 'zero_pad_detectors': 1=zero pad in lateral (X) direction; 0=don't typically 1
             ** 'zero_pad_time': 1=zero pad in axial (t,time) direction; 0=don't typically 1
             ** 'fourier_coefficients_dim': signal fourier coefficients a single image fourier coefficient is interploated (5)
             ** 'samplingX': 1,defines how many image lines are reconstructed per transducer element. For value>1, the additional image lines are equidistantly placed between the transducer elements
             ** 'spacing_m': the target resolution in meters. Default resolution is 0.1 mm.

         Returns
         ----------
         
         :return:
         """
        speed_of_sound_in_m_per_s = 1480
        if "speed_of_sound_m_s" in kwargs:
            speed_of_sound_in_m_per_s = kwargs["speed_of_sound_m_s"]

        delay = 0
        if "delay" in kwargs:
            delay = kwargs["delay"]

        zero_pad_detectors = 1
        if "zero_pad_detectors" in kwargs:
            zero_pad_detectors = kwargs["zero_pad_detectors"]

        zero_pad_time = 1
        if "zero_pad_time" in kwargs:
            zero_pad_time = kwargs["zero_pad_time"]

        coeffT = 5
        if "fourier_coefficients_dim" in kwargs:
            coeffT = kwargs["fourier_coefficients_dim"]

        spacing_m = 0.0001
        if "spacing_m" in kwargs:
            spacing_m = kwargs["spacing_m"]

        reconstruction = fft_based_jaeger_2d(time_series_data,
                                             detection_elements=detection_elements,
                                             sampling_rate=sampling_rate,
                                             sos=speed_of_sound_in_m_per_s,
                                             delay=delay,
                                             zero_pad_detectors=zero_pad_detectors,
                                             zero_pad_time=zero_pad_time,
                                             fourier_coefficients_dim=coeffT,
                                             spacing_m=spacing_m,
                                             field_of_view=field_of_view)

        return reconstruction


class FFTbasedHauptmann2018(ReconstructionAlgorithm):

    def get_name(self) -> str:
        return "FFT-based reconstruction (Hauptmann 2018)"

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       sampling_rate: float,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
        Implementation of an FFT-based reconstruction algorithm.

        The implementation reflects the reconstruction algorithm described by Hauptmann et al., 2018.
        Additionally, original papers are also cited:
        
            Hauptmann, A., Cox, B., Lucka, F., Huynh, N., Betcke, M., Beard, P., & Arridge, S. 
            (2018, September). Approximate k-space models and deep learning for fast photoacoustic 
            reconstruction. In International Workshop on Machine Learning for Medical Image Reconstruction 
            (pp. 103-111). Springer, Cham.

            Köstli, K. P., Frenz, M., Bebie, H., & Weber, H. P. (2001). Temporal backward projection 
            of optoacoustic pressure transients using Fourier transform methods. Physics in Medicine 
            & Biology, 46(7), 1863.
            
            Xu, Y., Feng, D., & Wang, L. V. (2002). Exact frequency-domain reconstruction for 
            thermoacoustic tomography. I. Planar geometry. IEEE transactions on medical imaging, 
            21(7), 823-828.

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

        reconstructed = fft_hauptmann_2d(time_series_data, detection_elements, sampling_rate,
                                         field_of_view, spacing_m, speed_of_sound_in_m_per_s)

        return reconstructed
