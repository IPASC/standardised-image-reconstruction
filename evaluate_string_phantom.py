# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT
# evaluation with a string phantom - mengjie 31102022

import simpa as sp
from simpa import Tags
from image_reconstruction.batch_reconstruction import reconstruct_ipasc_hdf5
from image_reconstruction.reconstruction_algorithms import BackProjection, DelayMultiplyAndSumAlgorithm, \
    FFTbasedHauptmann2018, FftBasedJaeger2007
import quality_assessment as qa
import matplotlib.pyplot as plt


# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager("path_config.env")

file_path = path_manager.get_hdf5_file_save_path() + "/" + "CompletePipelineExample_4711" + ".hdf5"
ipasc_hdf5 = path_manager.get_hdf5_file_save_path() + "/" + "CompletePipelineExample_4711"+ "_ipasc.hdf5"

initial_pressure = sp.load_data_field(file_path, Tags.DATA_FIELD_INITIAL_PRESSURE,wavelength=850).astype(float)

segmentation = sp.load_data_field(file_path, Tags.DATA_FIELD_SEGMENTATION, wavelength=850)

settings = {
            "spacing_m": 0.2/ 1000,
            "speed_of_sound_m_s": 1540,
            "lowcut": int(0.1e4),
            "highcut": int(8e6),
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
