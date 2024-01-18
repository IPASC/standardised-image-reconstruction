# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT

import os
import glob
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
import glob
import shutil

NAME = "SEGMENTATION_LOADER"
INPUT_MASK_PATHS = r"D:\image_recon_project\segmentation_data/"
OUT_OF_PLANE_VESSEL_CUTOFF = True

for folder in ["1", "2", "3", "4"]:
    input_folder = fr"{INPUT_MASK_PATHS}/{folder}/Labels/"
    save_folder = fr"{INPUT_MASK_PATHS}/{folder}/Sims/"
    all_inputs = glob.glob(f"{input_folder}/*.nrrd")
    for label_mask_input in all_inputs[0:3]:
        mask_name = label_mask_input.split("\\")[-1].split("/")[-1].split("-labels.nrrd")[0]

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

        label_mask, _ = nrrd.read(label_mask_input)
        label_mask = label_mask.reshape((label_mask.shape[0], 1, label_mask.shape[1]))

        first_pixel_tissue = [np.asarray(np.argwhere(label_mask[x, 0, :] == 3))[0].item() for x in range(0, label_mask.shape[0])]

        for idx, x in enumerate(range(0, label_mask.shape[0])):
            label_mask[x, :, first_pixel_tissue[idx]] = 2
            label_mask[x, :, first_pixel_tissue[idx] + 1] = 2
            label_mask[x, :, first_pixel_tissue[idx] + 2] = 2
            label_mask[x, :, first_pixel_tissue[idx] + 3] = 2

        xdim, ydim, zdim = int(np.round(dim_x_mm/spacing)), int(np.round(dim_y_mm/spacing)), int(np.round(dim_z_mm/spacing))
        input_spacing = 0.06946983546
        num_mask_voxels_y = int(np.round((dim_y_mm / spacing) * (spacing / input_spacing)))

        segmentation_volume_tiled = np.tile(label_mask, (1, num_mask_voxels_y, 1))

        if OUT_OF_PLANE_VESSEL_CUTOFF:
            segmentation_volume_tiled[:, 0:int(num_mask_voxels_y / 2 - 5), :][
                segmentation_volume_tiled[:, 0:int(num_mask_voxels_y / 2 - 5), :] == 4] = 3
            segmentation_volume_tiled[:, int(num_mask_voxels_y / 2 + 5):, :][
                segmentation_volume_tiled[:, int(num_mask_voxels_y / 2 + 5):, :] == 4] = 3

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
            ret_dict[2] = sp.TISSUE_LIBRARY.epidermis(melanosom_volume_fraction=0.005)
            ret_dict[3] = sp.TISSUE_LIBRARY.muscle(background_oxy=0.7, blood_volume_fraction=0.01)
            ret_dict[4] = sp.TISSUE_LIBRARY.blood(oxygenation=0.9)
            return ret_dict


        settings.set_volume_creation_settings({
            Tags.INPUT_SEGMENTATION_VOLUME: segmentation_volume_mask,
            Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),

        })

        acoustic_settings = settings.get_acoustic_settings()
        # For this simulation: Use the created absorption map as the input initial pressure
        acoustic_settings[Tags.DATA_FIELD] = Tags.DATA_FIELD_INITIAL_PRESSURE

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
                                                                      pitch_mm=0.315,
                                                                      number_detector_elements=128,
                                                                      sampling_frequency_mhz=40,
                                                                      field_of_view_extent_mm=np.asarray([-128*0.1575, 128*0.1575, 0, 0, 0, 40])))
        device.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[40, 0, 0]))

        sp.simulate(simulation_pipeline=pipeline,
                    settings=settings,
                    digital_device_twin=device)

        file_path = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + ".hdf5"
        ipasc_hdf5 = path_manager.get_hdf5_file_save_path() + "/" + settings[Tags.VOLUME_NAME] + "_ipasc.hdf5"

        shutil.move(file_path, f"{save_folder}/{mask_name}.hdf5")
        shutil.move(ipasc_hdf5, f"{save_folder}/{mask_name}_ipasc.hdf5")
