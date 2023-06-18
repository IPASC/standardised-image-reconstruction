# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VOLUME_TRANSDUCER_DIM_IN_MM = 30 #
VOLUME_PLANAR_DIM_IN_MM = 30
VOLUME_HEIGHT_IN_MM = 30
SPACING = 0.2
RANDOM_SEED = 4712 #4711

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True


def create_example_tissue():
    """
    This example creates a phantom with strings over a homogenous background
    """
    # Background is water
    # Parameters (mua=0.043 cm-1, mus=10.0 cm-1, g=1.0) are set according to its properties at 850 nm
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(0.043, 10.0, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    # add strings
    # parameters for strings (mua=1.3, mus=10, g=0.9) are constant
    num_strings_per_row = 4
    num_strings_per_col = 4
    strings_spacing = 5
    for i in range(0, num_strings_per_row):
        for j in range(0, num_strings_per_col):
            tissue_dict["vessels_%d_%d" % (j,i)]=sp.define_circular_tubular_structure_settings(
            tube_start_mm=[7.5 + strings_spacing * j, 0, 5 + int(strings_spacing * i)],
            tube_end_mm=[7.5 + strings_spacing * j, VOLUME_PLANAR_DIM_IN_MM, 5 + int(strings_spacing * i)],
            molecular_composition=sp.TISSUE_LIBRARY.constant(1.3, 10.0, 0.9),
            radius_mm=1, priority=1, consider_partial_volume=False,
            adhere_to_deformation=False
            )

    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

#np.random.seed(RANDOM_SEED)
#VOLUME_NAME = "CompletePipelineTestMSOT_"+str(RANDOM_SEED)

general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.RANDOM_SEED: RANDOM_SEED,
            Tags.VOLUME_NAME: "CompletePipelineExample_" + str(RANDOM_SEED),
            Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
            Tags.SPACING_MM: SPACING,
            Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
            Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
            Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
            Tags.VOLUME_CREATOR: Tags.VOLUME_CREATOR_VERSATILE,
            Tags.GPU: True,
            Tags.WAVELENGTHS: [850],
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

settings.set_reconstruction_settings({
    Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: False,
    Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    Tags.ACOUSTIC_SIMULATION_3D: False,
    Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
    Tags.TUKEY_WINDOW_ALPHA: 0.5,
    Tags.BANDPASS_CUTOFF_LOWPASS: int(8e6),
    Tags.BANDPASS_CUTOFF_HIGHPASS: int(0.1e4),
    Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: False,
    Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
    Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
    Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
    Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
    Tags.KWAVE_PROPERTY_PMLInside: False,
    Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
    Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
    Tags.KWAVE_PROPERTY_PlotPML: False,
    Tags.RECORDMOVIE: False,
    Tags.MOVIENAME: "visualization_log",
    Tags.ACOUSTIC_LOG_SCALE: True,
    Tags.DATA_FIELD_SPEED_OF_SOUND: 1540,
    Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
    Tags.DATA_FIELD_DENSITY: 1000,
    Tags.SPACING_MM: SPACING
})

settings["noise_initial_pressure"] = {
    Tags.NOISE_MEAN: 1,
    Tags.NOISE_STD: 0.01,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

settings["noise_time_series"] = {
    Tags.NOISE_STD: 1,
    Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
    Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA
}

# TODO: For the device choice, uncomment the undesired device

# device = sp.MSOTAcuityEcho(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
#                                                      VOLUME_PLANAR_DIM_IN_MM/2,
#                                                      0]))
# device.update_settings_for_use_of_model_based_volume_creator(settings)

device = sp.PhotoacousticDevice(device_position_mm=np.array([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                             VOLUME_PLANAR_DIM_IN_MM/2,
                                                             0]),
                                field_of_view_extent_mm=np.asarray([-12.5+0.125, 12.5-0.125, 0, 0, 0, 30]))
device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                              pitch_mm=0.25,
                                                              number_detector_elements=100,
                                                              field_of_view_extent_mm=np.asarray([-12.5+0.125, 12.5-0.125, 0, 0, 0, 30])))
print(device.get_detection_geometry().get_detector_element_positions_base_mm())
device.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[100, 0, 0]))


SIMULATION_PIPELINE = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MCXAdapter(settings),
    sp.GaussianNoise(settings, "noise_initial_pressure"),
    sp.KWaveAdapter(settings),
    sp.GaussianNoise(settings, "noise_time_series"),
    sp.TimeReversalAdapter(settings),
    sp.FieldOfViewCropping(settings)
    ]

sp.simulate(SIMULATION_PIPELINE, settings, device)

wavelength = settings[Tags.WAVELENGTHS][0]
file_path=path_manager.get_hdf5_file_save_path() + "/" + "CompletePipelineExample_" + str(RANDOM_SEED)+".hdf5"
segmentation_mask = sp.load_data_field(file_path=file_path,
                                       wavelength=wavelength,
                                       data_field=Tags.DATA_FIELD_SEGMENTATION)

reco = np.rot90(sp.load_data_field(file_path, wavelength=wavelength, data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA), -1)
time_series = np.rot90(sp.load_data_field(file_path, wavelength=wavelength, data_field=Tags.DATA_FIELD_TIME_SERIES_DATA), -1)
initial_pressure = np.rot90(sp.load_data_field(file_path, wavelength=wavelength, data_field=Tags.DATA_FIELD_INITIAL_PRESSURE), -1)

plt.figure(figsize=(7, 3))
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(initial_pressure)
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(time_series, aspect=0.18)
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(reco)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(SAVE_PATH, "result.svg"), dpi=300)
# plt.close()
