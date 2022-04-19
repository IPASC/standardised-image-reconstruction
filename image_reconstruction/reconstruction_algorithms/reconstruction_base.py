"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-License-Identifier: MIT
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from pacfish import PAData, load_data
from image_reconstruction.reconstruction_utils.post_processing import apply_post_processing
from image_reconstruction.reconstruction_utils.pre_processing import apply_pre_processing


class ReconstructionAlgorithm(ABC):
    """
    This class should be used as the base class for the implementation of
    photoacoustic image reconstruction algorithms.

    """
    def __init__(self):
        super(ReconstructionAlgorithm, self).__init__()
        self.ipasc_data: PAData = PAData()

    def reconstruct_time_series_data(self, path_to_ipasc_hdf5: str, **kwargs):
        """
        This method should be called to perform image reconstruction.
        It is independent of the actual algorithm implementation and performs input validation of the
        IPASC HDF5 file container and implements data loading.

        :param path_to_ipasc_hdf5: A string that
        :param kwargs:
        :return:
        """

        # Input validation with descriptive error messages
        if not os.path.exists(path_to_ipasc_hdf5):
            raise FileNotFoundError(f"The given file path ({path_to_ipasc_hdf5}) does not exist.")
        if not os.path.isfile(path_to_ipasc_hdf5):
            raise FileNotFoundError(f"The given file path ({path_to_ipasc_hdf5}) does not point to a file.")
        if not path_to_ipasc_hdf5.endswith(".hdf5"):
            raise AssertionError(f"The given file path must point to an hdf5 file that ends with '.hdf5'")

        # data loading
        self.ipasc_data = load_data(path_to_ipasc_hdf5)
        print("Sampling rate: ", self.ipasc_data.get_sampling_rate())
        field_of_view = self.ipasc_data.get_field_of_view()
        # TODO: if field of view is None, set a default field of view.
        print("Reconstructing in this field of view FOV [m]:", field_of_view)

        detection_elements = dict()
        detection_elements['positions'] = self.ipasc_data.get_detector_position()
        detection_elements['orientations'] = self.ipasc_data.get_detector_orientation()
        detection_elements['geometry'] = self.ipasc_data.get_detector_geometry()
        detection_elements['geometry_type'] = self.ipasc_data.get_detector_geometry_type()
        time_series_data = self.ipasc_data.binary_time_series_data

        if len(np.shape(time_series_data)) == 2:
            # Assume wavelengths and frame are singleton dimensions
            time_series_data = np.reshape(time_series_data, (np.shape(time_series_data)[0],
                                                             np.shape(time_series_data)[1],
                                                             1, 1))

        num_wavelengths = np.shape(time_series_data)[2]
        num_frames = np.shape(time_series_data)[3]
        wavelengths = []
        for wl_idx in range(num_wavelengths):
            frames = []
            for frame_idx in range(num_frames):
                ts_data = time_series_data[:, :, wl_idx, frame_idx]
                ts_data = apply_pre_processing(ts_data, self.ipasc_data.get_sampling_rate(), **kwargs)
                reconstruction = self.implementation(time_series_data=ts_data,
                                                     detection_elements=detection_elements,
                                                     field_of_view=field_of_view, **kwargs)
                reconstruction = apply_post_processing(reconstruction, **kwargs)
                frames.append(reconstruction)
            wavelengths.append(frames)

        result = np.moveaxis(np.asarray(wavelengths), [0, 1, 2, 3, 4], [3, 4, 0, 1, 2])
        return result

    @abstractmethod
    def implementation(self, time_series_data: np.ndarray, detection_elements: dict,
                       field_of_view: np.ndarray, **kwargs):
        """
        This method is extended by each class that represents one photoacoustic image reconstruction algorithm.

        :param time_series_data: A 4D numpy array with the following internal array definition:
                                [detectors, time samples]
        :param detection_elements: A dictionary that describes the detection geometry.
                                   The dictionary contains three entries:
                                   ** "positions": The positions of the detection elements relative to the field of view
                                   ** "orientations": The orientations of the detection elements
                                   ** "sizes": The sizes of the detection elements.
        :param field_of_view: A 1D 6 element-long numpy array that contains the extent of the field of view in x, y and
                              z direction in the same coordinate system as the detection element positions.
        :param kwargs: A dictionary containing any further parameters adjustible for the algorithm. Sensible defaults
                       are assumed for each of the parameters, such that the algorithm also runs if no kwargs
                       are given. The possible parameters are documented in the *implementation* method of the
                       subclass.
        :return: A 5D numpy array containing the reconstructed data with the following internal array representation:
                 [x samples, y samples, z samples, wavelength, frames]
        """

        pass
