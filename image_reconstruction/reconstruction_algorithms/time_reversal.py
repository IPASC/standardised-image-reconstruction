from image_reconstruction.reconstruction_algorithms import ReconstructionAlgorithm
from image_reconstruction.reconstruction_utils.beamforming.time_reversal import time_reversal_kwave_wrapper
import numpy as np


class TimeReversal(ReconstructionAlgorithm):

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
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
             ** 'spacing_m': the target resolution in meters. Default resolution is 0.1 mm.

         Returns
         ----------

         :return:
         """
        speed_of_sound_in_m_per_s = 1480
        if "speed_of_sound_m_s" in kwargs:
            speed_of_sound_in_m_per_s = kwargs["speed_of_sound_m_s"]

        spacing_m = 0.0001
        if "spacing_m" in kwargs:
            spacing_m = kwargs["spacing_m"]

        reconstruction = time_reversal_kwave_wrapper(time_series_data,
                                                     detection_elements=detection_elements,
                                                     sampling_rate=self.ipasc_data.get_sampling_rate(),
                                                     sos=speed_of_sound_in_m_per_s,
                                                     spacing_m=spacing_m,
                                                     field_of_view=field_of_view)

        return reconstruction
