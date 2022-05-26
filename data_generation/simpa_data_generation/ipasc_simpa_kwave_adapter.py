# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT

import os
import inspect
import subprocess
import numpy as np
import simpa as sp
import scipy.io as sio
from scipy.spatial.transform import Rotation
from simpa.utils.calculate import rotation_matrix_between_vectors
from simpa.core.simulation_modules.acoustic_forward_module import AcousticForwardModelBaseAdapter


class IpascSimpaKWaveAdapter(AcousticForwardModelBaseAdapter):
    """

    """

    def forward_model(self, detection_geometry) -> np.ndarray:

        # Load Tags.DATA_FIELD data and export it as a .mat file
        data_dict = {}
        mat_file_path = self.global_settings[sp.Tags.SIMULATION_PATH] + "/" + \
                        self.global_settings[sp.Tags.VOLUME_NAME] + \
                        "_kwave.mat"

        tmp_ac_data = sp.load_data_field(self.global_settings[sp.Tags.SIMPA_OUTPUT_PATH],
                                         sp.Tags.SIMULATION_PROPERTIES,
                                         self.global_settings[sp.Tags.WAVELENGTH])

        # Load initial pressure (or initial pressure proxy as defined by sp.tags.DATA_FIELD)
        initial_pressure = sp.load_data_field(self.global_settings[sp.Tags.SIMPA_OUTPUT_PATH],
                                              self.component_settings[sp.Tags.DATA_FIELD],
                                              self.global_settings[sp.Tags.WAVELENGTH])

        # Load the device definition (copied from SIMPA k-Wave adapter)
        detector_positions_mm = detection_geometry.get_detector_element_positions_accounting_for_device_position_mm()
        detector_positions_mm = np.moveaxis(detector_positions_mm, 1, 0)
        data_dict[sp.Tags.SENSOR_ELEMENT_POSITIONS] = detector_positions_mm[[2, 1, 0]]
        orientations = detection_geometry.get_detector_element_orientations()
        x_angles = np.arccos(np.dot(orientations, np.array([1, 0, 0]))) * 360 / (2 * np.pi)
        y_angles = np.arccos(np.dot(orientations, np.array([0, 1, 0]))) * 360 / (2 * np.pi)
        z_angles = np.arccos(np.dot(orientations, np.array([0, 0, 1]))) * 360 / (2 * np.pi)
        intrinsic_euler_angles = list()
        for orientation_vector in orientations:
            mat = rotation_matrix_between_vectors(orientation_vector, np.array([0, 0, 1]))
            rot = Rotation.from_matrix(mat)
            euler_angles = rot.as_euler("XYZ", degrees=True)
            intrinsic_euler_angles.append(euler_angles)
        intrinsic_euler_angles.reverse()
        angles = np.array([z_angles[::-1], y_angles[::-1], x_angles[::-1]])
        data_dict[sp.Tags.KWAVE_PROPERTY_DIRECTIVITY_ANGLE] = angles
        data_dict[sp.Tags.KWAVE_PROPERTY_INTRINSIC_EULER_ANGLE] = intrinsic_euler_angles

        # export speed of sound
        axes = (0, 2)
        image_slice = np.s_[:]
        data_dict[sp.Tags.DATA_FIELD_SPEED_OF_SOUND] = np.rot90(tmp_ac_data[sp.Tags.DATA_FIELD_SPEED_OF_SOUND]
                                                                [image_slice], 3, axes=axes)
        data_dict[sp.Tags.DATA_FIELD_DENSITY] = np.rot90(tmp_ac_data[sp.Tags.DATA_FIELD_DENSITY]
                                                         [image_slice], 3, axes=axes)
        data_dict[sp.Tags.DATA_FIELD_ALPHA_COEFF] = np.rot90(tmp_ac_data[sp.Tags.DATA_FIELD_ALPHA_COEFF]
                                                             [image_slice], 3, axes=axes)

        # Export settings
        data_dict[sp.Tags.DATA_FIELD_INITIAL_PRESSURE] = initial_pressure

        # Save export dict to mat file
        sio.savemat(mat_file_path, data_dict, long_field_names=True)

        # get current file path relative to this py file.
        base_script_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        # call matlab binary
        cmd = list()
        cmd.append(self.component_settings[sp.Tags.ACOUSTIC_MODEL_BINARY_PATH])
        cmd.append("-nodisplay")
        cmd.append("-nosplash")
        cmd.append("-automation")
        cmd.append("-wait")
        cmd.append("-r")
        cmd.append("addpath('" + base_script_path + "/');" +
                   "addpath('" + base_script_path + "/../../../PACFISH/pacfish_matlab/');" +
                   "ipasc_linear_array_simulation" + "('" + mat_file_path + "');exit;")
        cur_dir = os.getcwd()
        self.logger.info(cmd)
        subprocess.run(cmd)

        raw_time_series_data = sio.loadmat(mat_file_path)[sp.Tags.DATA_FIELD_TIME_SERIES_DATA]

        # delete exported data

        if os.path.exists(mat_file_path):
            os.remove(mat_file_path)
        if os.path.exists(mat_file_path + "dt.mat"):
            os.remove(mat_file_path + "dt.mat")
        os.chdir(cur_dir)

        return raw_time_series_data