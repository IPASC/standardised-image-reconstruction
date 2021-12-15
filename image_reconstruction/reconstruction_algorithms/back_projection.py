"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""
import numpy as np
from image_reconstruction.reconstruction_utils.beamforming import back_projection
from image_reconstruction.reconstruction_algorithms import ReconstructionAlgorithm
from image_reconstruction.reconstruction_utils.pre_processing import butter_bandpass_filter
from image_reconstruction.reconstruction_utils.post_processing import hilbert_transform_1_d
from image_reconstruction.reconstruction_utils.post_processing import log_compression


class BackProjection(ReconstructionAlgorithm):

    def implementation(self, time_series_data: np.ndarray,
                       detection_elements: dict,
                       field_of_view: np.ndarray,
                       **kwargs):
        """
        Implementation of a baseline delay and sum algorithm without any additional features.

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

        spacing_m = 0.0005
        if "spacing_m" in kwargs:
            spacing_m = kwargs["spacing_m"]

        lowcut = None
        if "lowcut" in kwargs:
            lowcut = kwargs["lowcut"]

        highcut = None
        if "highcut" in kwargs:
            highcut = kwargs["highcut"]

        filter_order = 5
        if "filter_order" in kwargs:
            filter_order = kwargs["filter_order"]

        envelope = False
        if "envelope" in kwargs:
            envelope = kwargs["envelope"]

        envelope_type = None
        if "envelope_type" in kwargs:
            envelope_type = kwargs["envelope_type"]

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

        if lowcut is not None or highcut is not None:
            time_series_data = butter_bandpass_filter(signal=time_series_data,
                                                      sampling_rate=self.ipasc_data.get_sampling_rate(),
                                                      lowcut=lowcut,
                                                      highcut=highcut,
                                                      order=filter_order)

        reconstructed = back_projection(time_series_data, detection_elements, self.ipasc_data.get_sampling_rate(),
                                        field_of_view, spacing_m, speed_of_sound_in_m_per_s,
                                        fnumber, p_scf, p_factor, p_pcf)

        if envelope:
            if envelope_type == "hilbert":
                # hilbert transform
                reconstructed = hilbert_transform_1_d(reconstructed, axis=0)
            elif envelope_type == "log":
                # hilbert transform + log-compression on 40 dB
                reconstructed = log_compression(reconstructed, axis=0, dynamic=40)
            elif envelope_type == "zero":
                # zero forcing
                reconstructed[reconstructed < 0] = 0
            elif envelope_type == "abs":
                # absolute value
                reconstructed = np.abs(reconstructed)
            else:
                print("WARN: No envelope type specified!")

        return reconstructed
