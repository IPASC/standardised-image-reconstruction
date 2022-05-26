# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags
import numpy as np
from utils.settings import generate_base_settings
from ipasc_simpa_kwave_adapter import IpascSimpaKWaveAdapter
from phantom_designs.phantom001 import phantom001

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

settings = generate_base_settings(path_manager)

# extract information on geometry and spacing for the purposes of volume creation and device definition
dim_x_mm = settings[Tags.DIM_VOLUME_X_MM]
dim_y_mm = settings[Tags.DIM_VOLUME_Y_MM]
dim_z_mm = settings[Tags.DIM_VOLUME_Z_MM]
spacing = settings[Tags.SPACING_MM]

# ###################################################################################
# VOLUME CREATION
#
# Case 1: using the SIMPA volume creation module
#
# ###################################################################################
settings.set_volume_creation_settings(phantom001())

acoustic_settings = settings.get_acoustic_settings()
# For this simulation: Use the created absorption map as the input initial pressure
acoustic_settings[Tags.DATA_FIELD] = Tags.DATA_FIELD_ABSORPTION_PER_CM

pipeline = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    IpascSimpaKWaveAdapter(settings),
    sp.TimeReversalAdapter(settings),
    sp.FieldOfViewCropping(settings)
]

# Create a device with
device = sp.PhotoacousticDevice(device_position_mm=np.array([dim_x_mm/2,
                                                             dim_y_mm/2,
                                                             0]),
                                field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20]))
device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                              pitch_mm=0.25,
                                                              number_detector_elements=100,
                                                              field_of_view_extent_mm=np.asarray([-15, 15, 0, 0, 0, 20])))
device.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[100, 0, 0]))

sp.simulate(simulation_pipeline=pipeline,
            settings=settings,
            digital_device_twin=device)
