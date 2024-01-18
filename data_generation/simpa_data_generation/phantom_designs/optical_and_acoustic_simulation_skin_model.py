# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-FileCopyrightText: 2022 Mengjie Shi
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_TRANSDUCER_DIM_IN_MM = 30
VOLUME_PLANAR_DIM_IN_MM = 30
VOLUME_HEIGHT_IN_MM = 30
SPACING = 0.2
# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.
RANDOM_SEED = 4713

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True

def create_example_tissue():
    """
    This example creates a skin phantom.
    It contains an ultrasound gel layer (2 mm) on top of an epidermis layer (0.4 mm), followed by a dermis layer (4mm),
    and a muscle layer.
    2 blood vessels are randomly distributed within the muscle layer.
    """

    tissue_dict = sp.Settings()
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(0.001, 1.0, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND
    background_dictionary[Tags.PRIORITY] = 0
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    tissue_dict["epidermis"] = sp.define_horizontal_layer_structure_settings(z_start_mm=2, thickness_mm=0.4,
                                                                             molecular_composition=
                                                                             sp.TISSUE_LIBRARY.constant(0.5, 10, 0.9),
                                                                             consider_partial_volume=False,
                                                                             adhere_to_deformation=False, priority=12)
    tissue_dict["dermis"] = sp.define_horizontal_layer_structure_settings(z_start_mm=2.4, thickness_mm=4,
                                                                          molecular_composition=
                                                                          sp.TISSUE_LIBRARY.constant(0.05, 10, 0.9),
                                                                          consider_partial_volume=False,
                                                                          adhere_to_deformation=False, priority=2)
    tissue_dict["muscle"] = sp.define_horizontal_layer_structure_settings(z_start_mm=5.4, thickness_mm=30,
                                                                          molecular_composition=
                                                                          sp.TISSUE_LIBRARY.constant(0.05, 10, 0.9),
                                                                          consider_partial_volume=False,
                                                                          adhere_to_deformation=False, priority=2)
    tissue_dict["vessel_1"] = sp.define_circular_tubular_structure_settings(
            tube_start_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/2 - 5, 0, 7],
            tube_end_mm=[VOLUME_TRANSDUCER_DIM_IN_MM/2 - 5, VOLUME_PLANAR_DIM_IN_MM, 7],
            molecular_composition=sp.TISSUE_LIBRARY.constant(10, 10, 0.9),
            radius_mm=1, priority=12)
    tissue_dict["vessel_2"] = sp.define_circular_tubular_structure_settings(
            tube_start_mm=[VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 5, 0, 9],
            tube_end_mm=[VOLUME_TRANSDUCER_DIM_IN_MM / 2 + 5, VOLUME_PLANAR_DIM_IN_MM, 9],
            molecular_composition=sp.TISSUE_LIBRARY.constant(10, 10, 0.9),
            radius_mm=2, priority=12)
    return tissue_dict


general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: RANDOM_SEED,
            Tags.VOLUME_NAME: "SkinModel_" + str(RANDOM_SEED),
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: SPACING,
            Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.GPU: True,
            Tags.WAVELENGTHS: [800],
            Tags.DO_FILE_COMPRESSION: True,
            Tags.DO_IPASC_EXPORT: True
        }
settings = sp.Settings(general_settings)
np.random.seed(RANDOM_SEED)

settings.set_volume_creation_settings({
    Tags.STRUCTURES: create_example_tissue(),
    Tags.SIMULATE_DEFORMED_LAYERS: True
})

settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.ILLUMINATION_TYPE: Tags.ILLUMINATION_TYPE_SLIT,
    Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    Tags.MCX_ASSUMED_ANISOTROPY: 0.9,
})

settings.set_acoustic_settings({
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.DATA_FIELD_SPEED_OF_SOUND: 1540, 
    Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
    Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
    Tags.KWAVE_PROPERTY_PMLInside: False,
    Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
    Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
    Tags.KWAVE_PROPERTY_PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True
})


device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                             VOLUME_PLANAR_DIM_IN_MM/2,
                                                             0]),
                                field_of_view_extent_mm=np.asarray([-12.5+0.125, 12.5-0.125, 0, 0, 0, 30]))
device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                              pitch_mm=0.25,
                                                              number_detector_elements=100,
                                                              field_of_view_extent_mm=np.asarray([-12.5+0.125, 12.5-0.125, 0, 0, 0, 30])))
device.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[30, 0, 0]))

SIMULATION_PIPELINE = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MCXAdapter(settings),
    sp.KWaveAdapter(settings),
    sp.FieldOfViewCropping(settings)
    ]

sp.simulate(SIMULATION_PIPELINE, settings, device)

wavelength = settings[Tags.WAVELENGTHS][0]
file_path=path_manager.get_hdf5_file_save_path() + "/" + "SkinModel_" + str(RANDOM_SEED)+".hdf5"
segmentation_mask = sp.load_data_field(file_path=file_path,
                                       wavelength=wavelength,
                                       data_field=Tags.DATA_FIELD_SEGMENTATION)

time_series = np.rot90(sp.load_data_field(file_path, wavelength=wavelength, data_field=Tags.DATA_FIELD_TIME_SERIES_DATA), -1)
initial_pressure = np.rot90(sp.load_data_field(file_path, wavelength=wavelength, data_field=Tags.DATA_FIELD_INITIAL_PRESSURE), -1)

plt.figure(figsize=(7, 3))
plt.subplot(1, 2, 1)
plt.axis('off')
plt.imshow(np.log(initial_pressure))
plt.subplot(1, 2, 2)
plt.axis('off')
plt.imshow(time_series, aspect=0.18)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(SAVE_PATH, "result.svg"), dpi=300)
# plt.close()
