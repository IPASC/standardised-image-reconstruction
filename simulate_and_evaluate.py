# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags
import numpy as np
from data_generation.simpa_data_generation.utils.settings import generate_base_settings
from data_generation.simpa_data_generation.ipasc_simpa_kwave_adapter import IpascSimpaKWaveAdapter
from data_generation.simpa_data_generation.phantom_designs import phantom_random_vessels as phantom
from image_reconstruction.batch_reconstruction import reconstruct_ipasc_hdf5
from image_reconstruction.reconstruction_algorithms import BackProjection, DelayMultiplyAndSumAlgorithm, \
    FFTbasedHauptmann2018, FftBasedJaeger2007
import quality_assessment as qa
import matplotlib.pyplot as plt
from random import choice

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager("path_config.env")

settings = generate_base_settings(path_manager, volume_name="e44d831e-0eb8-4734-81d7-e399a255e0c3")

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
settings.set_volume_creation_settings(phantom(dim_x_mm, dim_y_mm, dim_z_mm))

acoustic_settings = settings.get_acoustic_settings()
# For this simulation: Use the created absorption map as the input initial pressure
acoustic_settings[Tags.DATA_FIELD] = Tags.DATA_FIELD_ABSORPTION_PER_CM

pipeline = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    IpascSimpaKWaveAdapter(settings),
    sp.FieldOfViewCropping(settings, "FieldOfViewCropping")
]

# Create a device with
device = sp.PhotoacousticDevice(device_position_mm=np.array([dim_x_mm/2,
                                                             dim_y_mm/2,
                                                             0]),
                                field_of_view_extent_mm=np.asarray([-128*0.15, 128*0.15, 0, 0, 0, 40]))
device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                              pitch_mm=0.3,
                                                              number_detector_elements=128,
                                                              sampling_frequency_mhz=50,
                                                              field_of_view_extent_mm=np.asarray([-128*0.15, 128*0.15, 0, 0, 0, 40])))
device.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[100, 0, 0]))

sp.simulate(simulation_pipeline=pipeline,
            settings=settings,
            digital_device_twin=device)

sp.visualise_data(settings=settings,
                  path_manager=path_manager,
                  wavelength=800,
                  show_absorption=True,
                  show_time_series_data=True,
                  show_segmentation_map=True)

file_path = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + ".hdf5"
ipasc_hdf5 = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + "_ipasc.hdf5"

initial_pressure = sp.load_data_field(file_path, acoustic_settings[Tags.DATA_FIELD], settings[Tags.WAVELENGTHS][0]).astype(float)

segmentation = sp.load_data_field(file_path, Tags.DATA_FIELD_SEGMENTATION, settings[Tags.WAVELENGTHS][0])

settings = {
            "spacing_m": settings[Tags.SPACING_MM] / 1000,
            "speed_of_sound_m_s": settings[Tags.DATA_FIELD_SPEED_OF_SOUND],
            "lowcut": None,
            "highcut": None,
            "order": 9,
            "envelope": True,
            "p_factor": 1,
            "p_SCF": 1,
            "p_PCF": 0,
            "fnumber": 0,
            "envelope_type": "abs",
            "delay": 0,
            "zeroX": 0,
            "zeroT": 0,
            "fourier_coefficients_dim": 5,
        }
algorithms = [(BackProjection(), settings),
              (DelayMultiplyAndSumAlgorithm(), settings),
              (FftBasedJaeger2007(), settings),
              (FFTbasedHauptmann2018(), settings)]

reconstructions = reconstruct_ipasc_hdf5(ipasc_hdf5, algorithms)

full_reference_measures = [qa.RootMeanSquaredError(),
                           qa.UniversalQualityIndex(),
                           qa.StructuralSimilarityIndex(),
                           qa.MutualInformation()]

no_reference_measures = [qa.GeneralisedSignalToNoiseRatio()]

plt.figure()
plt.subplot(1, len(algorithms)+1, 1)
plt.title("Ground Truth")
plt.imshow(initial_pressure.T)
plt.colorbar()

index = 2
for (algorithm, settings), reconstruction in zip(algorithms, reconstructions):
    plt.subplot(1, len(algorithms)+1, index)
    print(algorithm.get_name())
    reconstruction = reconstruction[:, 0, :, 0, 0].astype(float)
    plt.title(algorithm.get_name())
    plt.imshow(reconstruction.T)
    plt.colorbar()
    for measure in full_reference_measures:
        print(measure.get_name(), measure.compute_measure(initial_pressure, reconstruction))

    for measure in no_reference_measures:
        print(measure.get_name(), measure.compute_measure(reconstruction, segmentation > 0, segmentation < 0))
    index += 1

plt.show()
plt.close()
