# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT

import os
import simpa as sp
import numpy as np
import pacfish as pf
from simpa import Tags
import quality_assessment as qa
import matplotlib.pyplot as plt
from data_generation.simpa_data_generation.utils.settings import generate_base_settings
from data_generation.simpa_data_generation.ipasc_simpa_kwave_adapter import IpascSimpaKWaveAdapter
from data_generation.simpa_data_generation.phantom_designs import phantom001 as phantom
from image_reconstruction.batch_reconstruction import reconstruct_ipasc_hdf5
from image_reconstruction.reconstruction_algorithms import BackProjection, DelayMultiplyAndSumAlgorithm, \
    FftBasedJaeger2007

NAME = "FULL_BANDWIDTH"

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager("path_config.env")

settings = generate_base_settings(path_manager, volume_name=NAME)

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

file_path = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + ".hdf5"
ipasc_hdf5 = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + "_ipasc.hdf5"
ipasc_hdf5_noise = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + "noise_ipasc.hdf5"

if not os.path.exists(ipasc_hdf5_noise):
    pa_data = pf.load_data(ipasc_hdf5)
    pa_data.binary_time_series_data = np.random.normal(0, 0.2*np.max(pa_data.binary_time_series_data)) + \
                                       pa_data.binary_time_series_data
    pa_data.binary_time_series_data = np.random.normal(1.0, 0.2, pa_data.binary_time_series_data.shape) * \
                                       pa_data.binary_time_series_data
    pf.write_data(ipasc_hdf5_noise, pa_data)

initial_pressure = sp.load_data_field(file_path, acoustic_settings[Tags.DATA_FIELD],
                                      settings[Tags.WAVELENGTHS][0]).astype(float)

segmentation = sp.load_data_field(file_path, Tags.DATA_FIELD_SEGMENTATION, settings[Tags.WAVELENGTHS][0])

settings = {
            "spacing_m": settings[Tags.SPACING_MM] / 1000,
            "speed_of_sound_m_s": settings[Tags.DATA_FIELD_SPEED_OF_SOUND],
            "lowcut": 1e4,
            "highcut": 2e7,
            "order": 9,
            "envelope": True,
            "p_factor": 1,
            "p_SCF": 1,
            "p_PCF": 0,
            "fnumber": 0,
            "envelope_type": "hilbert",
            "delay": 0,
            "zeroX": 0,
            "zeroT": 0,
            "fourier_coefficients_dim": 5,
            "scaling_method": "mean"
        }
algorithms = [(BackProjection(), settings),
              (DelayMultiplyAndSumAlgorithm(), settings),
              (FftBasedJaeger2007(), settings)]

reconstructions = reconstruct_ipasc_hdf5(ipasc_hdf5, algorithms)
recons_noise = reconstruct_ipasc_hdf5(ipasc_hdf5_noise, algorithms)

full_reference_measures = [qa.RootMeanSquaredError(),
                           qa.UniversalQualityIndex(),
                           qa.MutualInformation(),
                           qa.StructuralSimilarityIndexTorch()]

no_reference_measures = [qa.GeneralisedSignalToNoiseRatio()]

plt.figure(figsize=(16, 6))
plt.subplot(2, len(algorithms)+1, 1)
plt.title("Ground Truth")
plt.imshow(initial_pressure.T)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel(f"RMSE: {full_reference_measures[0].compute_measure(initial_pressure, initial_pressure):.3f}\n"
           f"SSIM: {full_reference_measures[-1].compute_measure(initial_pressure, initial_pressure):.3f}\n"
           f"gCNR: {no_reference_measures[0].compute_measure(initial_pressure, segmentation > 0, segmentation < 0):.3f}")
cbar = plt.colorbar()
cbar.set_label("Initial Pressure [a.u.]")

index = 2
for (algorithm, settings), reconstruction in zip(algorithms, reconstructions):
    plt.subplot(2, len(algorithms)+1, index)
    reconstruction = reconstruction[:, 0, :, 0, 0].astype(float)
    plt.title(algorithm.get_name())
    plt.imshow(reconstruction.T)
    plt.xlabel(f"RMSE: {full_reference_measures[0].compute_measure(initial_pressure, reconstruction):.3f}\n"
               f"SSIM: {full_reference_measures[-1].compute_measure(initial_pressure, reconstruction):.3f}\n"
               f"gCNR: {no_reference_measures[0].compute_measure(reconstruction, segmentation > 0, segmentation < 0):.3f}")
    plt.xticks([], [])
    plt.yticks([], [])
    cbar = plt.colorbar()
    cbar.set_label("Signal [a.u.]")
    index += 1

plt.subplot(2, len(algorithms)+1, index)
plt.title("With 20% noise")
index +=1
for (algorithm, settings), reconstruction in zip(algorithms, recons_noise):
    plt.subplot(2, len(algorithms)+1, index)
    reconstruction = reconstruction[:, 0, :, 0, 0].astype(float)
    plt.title(algorithm.get_name())
    plt.imshow(reconstruction.T)
    plt.xlabel(f"RMSE: {full_reference_measures[0].compute_measure(initial_pressure, reconstruction):.3f}\n"
               f"SSIM: {full_reference_measures[-1].compute_measure(initial_pressure, reconstruction):.3f}\n"
               f"gCNR: {no_reference_measures[0].compute_measure(reconstruction, segmentation > 0, segmentation < 0):.3f}")
    plt.xticks([], [])
    plt.yticks([], [])
    cbar = plt.colorbar()
    cbar.set_label("Signal [a.u.]")
    index += 1

plt.tight_layout()
plt.savefig(f"{NAME}.png", dpi=300)
plt.close()
