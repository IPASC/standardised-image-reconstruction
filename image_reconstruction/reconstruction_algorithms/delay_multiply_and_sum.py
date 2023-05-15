"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)

Please note that the code here is an adapted version of the code
published in the SIMPA repository also under the MIT license:
https://github.com/IMSY-DKFZ/simpa

SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
SPDX-FileCopyrightText: Janek Groehl
SPDX-License-Identifier: MIT
"""
import numpy as np
from image_reconstruction.reconstruction_algorithms import ReconstructionAlgorithm
from image_reconstruction.reconstruction_utils.beamforming import delay_multiply_and_sum


class DelayMultiplyAndSumAlgorithm(ReconstructionAlgorithm):

    def get_name(self) -> str:
        return "Delay-Multiply-And-Sum"

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       sampling_rate: float,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
        Implementation of the delay-multiply-and-sum algorithm.

        The implementation is based on the original publication for the application to ultrasound images by Matrone et al::

            Matrone, Giulia, Alessandro Stuart Savoia, Giosuè Caliano, and Giovanni Magenes. 
            "The delay multiply and sum beamforming algorithm in ultrasound B-mode medical imaging."
            IEEE transactions on medical imaging 34, no. 4 (2014): 940-949.

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

            ** 'fnumber' the fnumber as described in this paper::

                    Perrot, Vincent, et al.
                    "So you think you can DAS?
                    A viewpoint on delay-and-sum beamforming." Ultrasonics 111 (2021): 106309.

            ** 'signed_dmas' boolean to define whether the sign of the original DAS is retained according to::

                    Kirchner, Thomas et al.
                    "Signed Real-Time Delay Multiply and Sum Beamforming for Multispectral Photoacoustic Imaging"
                    J. Imaging 2018, 4(10), 121

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

        fnumber = 0
        if "fnumber" in kwargs:
            fnumber = kwargs["fnumber"]

        signed_dmas = False
        if "signed_dmas" in kwargs:
            signed_dmas = kwargs["signed_dmas"]

        reconstructed = delay_multiply_and_sum(time_series_data=time_series_data,
                                               detection_elements=detection_elements,
                                               sampling_rate=sampling_rate,
                                               field_of_view=field_of_view,
                                               spacing_m=spacing_m,
                                               speed_of_sound_in_m_per_s=speed_of_sound_in_m_per_s,
                                               fnumber=fnumber,
                                               signed_dmas=signed_dmas)

        return reconstructed
