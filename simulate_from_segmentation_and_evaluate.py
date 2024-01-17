# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT

import os
import nrrd
import simpa as sp
import numpy as np
import pacfish as pf
from scipy.ndimage import zoom
from simpa import Tags
import quality_assessment as qa
import matplotlib.pyplot as plt
from data_generation.simpa_data_generation.utils.settings import generate_base_settings
from data_generation.simpa_data_generation.ipasc_simpa_kwave_adapter import IpascSimpaKWaveAdapter
from data_generation.simpa_data_generation.phantom_designs import phantom001 as phantom
from image_reconstruction.batch_reconstruction import reconstruct_ipasc_hdf5
from image_reconstruction.reconstruction_algorithms import BackProjection, DelayMultiplyAndSumAlgorithm, \
    FftBasedJaeger2007

NAME = "SEGMENTATION_LOADER"
INPUT_MASK_PATH = "segmentations/finger_model/1-PA-labels.nrrd"
# INPUT_MASK_PATH = "segmentations/human_forearm_model/1-PA-labels.nrrd"


# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager("C:/Users/grohl01/path_config.env")

settings = generate_base_settings(path_manager, volume_name=NAME)

# extract information on geometry and spacing for the purposes of volume creation and device definition
dim_x_mm = settings[Tags.DIM_VOLUME_X_MM]
dim_y_mm = settings[Tags.DIM_VOLUME_Y_MM]
dim_z_mm = settings[Tags.DIM_VOLUME_Z_MM]
spacing = settings[Tags.SPACING_MM]

# ###################################################################################
# VOLUME CREATION
#
# Case 2: Using the segmentation loader
#
# ###################################################################################

label_mask, _ = nrrd.read(INPUT_MASK_PATH)
label_mask = label_mask.reshape((label_mask.shape[0], 1, label_mask.shape[1]))

first_pixel_tissue = [np.asarray(np.argwhere(label_mask[x, 0, :] == 3))[0].item() for x in range(0, label_mask.shape[0])]

for idx, x in enumerate(range(0, label_mask.shape[0])):
    label_mask[x, :, first_pixel_tissue[idx]] = 2
    label_mask[x, :, first_pixel_tissue[idx]+1] = 2
    label_mask[x, :, first_pixel_tissue[idx]+2] = 2

xdim, ydim, zdim = int(np.round(dim_x_mm/spacing)), int(np.round(dim_y_mm/spacing)), int(np.round(dim_z_mm/spacing))
input_spacing = 0.06946983546
num_mask_voxels_y = int(np.round((dim_y_mm / spacing) * (spacing / input_spacing)))

segmentation_volume_tiled = np.tile(label_mask, (1, num_mask_voxels_y, 1))
segmentation_volume = np.round(zoom(segmentation_volume_tiled, input_spacing/spacing,
                                         order=0)).astype(int)
segmentation_volume_mask = np.ones((xdim, ydim, zdim))
x_offset = int((xdim-segmentation_volume.shape[0]) / 2)
y_offset = int((ydim-segmentation_volume.shape[1]) / 2)
z_offset = int((zdim-segmentation_volume.shape[2]) / 2)
segmentation_volume_mask[x_offset:x_offset+segmentation_volume.shape[0],
                         y_offset:y_offset+segmentation_volume.shape[1],
                         0:segmentation_volume.shape[2]] = segmentation_volume

def segmentation_class_mapping():
    ret_dict = dict()
    ret_dict[0] = sp.MolecularCompositionGenerator().append(sp.MoleculeLibrary().water()).get_molecular_composition(sp.SegmentationClasses.WATER)
    ret_dict[1] = sp.MolecularCompositionGenerator().append(sp.MoleculeLibrary().water()).get_molecular_composition(sp.SegmentationClasses.WATER)
    ret_dict[2] = sp.TISSUE_LIBRARY.epidermis()
    ret_dict[3] = sp.TISSUE_LIBRARY.muscle(background_oxy=0.7, blood_volume_fraction=0.1)
    ret_dict[4] = sp.TISSUE_LIBRARY.blood(oxygenation=0.9)
    return ret_dict


settings.set_volume_creation_settings({
    Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
    Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),

})

acoustic_settings = settings.get_acoustic_settings()
# For this simulation: Use the created absorption map as the input initial pressure
acoustic_settings[Tags.DATA_FIELD] = Tags.DATA_FIELD_ABSORPTION_PER_CM

pipeline = [
    sp.SegmentationBasedVolumeCreationAdapter(settings),
    sp.MCXAdapter(settings),
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

# sp.simulate(simulation_pipeline=pipeline,
#             settings=settings,
#             digital_device_twin=device)

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
            "p_factor": 1,
            "p_SCF": 1,
            "p_PCF": 0,
            "fnumber": 0,
            "non_negativity_method": "hilbert",
            "delay": 0,
            "zeroX": 0,
            "zeroT": 0,
            "fourier_coefficients_dim": 5,
            "scaling_method": "mean",
            "apodisation": "hamming"
        }
algorithms = [(BackProjection(), settings),
              (DelayMultiplyAndSumAlgorithm(), settings),
              (FftBasedJaeger2007(), settings)]

reconstructions = reconstruct_ipasc_hdf5(ipasc_hdf5, algorithms)
recons_noise = reconstruct_ipasc_hdf5(ipasc_hdf5_noise, algorithms)

full_reference_measures = [qa.RootMeanSquaredError(),
                           qa.UniversalQualityIndex(),
                           qa.MutualInformation(),
                           qa.StructuralSimilarityIndex()]

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
