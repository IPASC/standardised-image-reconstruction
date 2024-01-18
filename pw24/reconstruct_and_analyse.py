import pacfish as pf
import numpy as np
import matplotlib.pyplot as plt
import simpa as sp
from image_reconstruction.reconstruction_algorithms import BackProjection, DelayMultiplyAndSumAlgorithm, FftBasedJaeger2007
from image_reconstruction.batch_reconstruction import reconstruct_ipasc_hdf5

#PATH = r"D:\image_recon_project\finger_data\4\Sims/1-PA_ipasc.hdf5"
#SPACING = 0.0390625 * 2 / 1000
PATH = r"D:\image_recon_project\in_silico_data\StringPhantom_4712_ipasc.hdf5"
SPACING = 0.2 / 1000
Y_UPPER = 65
Y_LOWER = 5

pa_data = pf.load_data(PATH)


initial_pressure = sp.load_data_field(PATH.replace("_ipasc", ""), sp.Tags.DATA_FIELD_INITIAL_PRESSURE,
                                      800).astype(float)

segmentation = sp.load_data_field(PATH.replace("_ipasc", ""),
                                  sp.Tags.DATA_FIELD_SEGMENTATION,
                                  800)

settings = {
            "spacing_m": SPACING,
            "speed_of_sound_m_s": 1540,
            "lowcut": 1e4,
            "highcut": 2e7,
            "order": 3,
            "non_negativity_method": "hilbert",
            "p_factor": 1,
            "p_SCF": 1,
            "p_PCF": 0,
            "fnumber": 0,
            "apodisation": "box",
            "time_interpolation_factor": 1,
            "detector_interpolation_factor": 1,
            "delay": 0,
            "zeroX": 0,
            "zeroT": 0,
            "fourier_coefficients_dim": 5,
            "scaling_method": "mean"
        }
algorithms = [(BackProjection(), settings),
              (DelayMultiplyAndSumAlgorithm(), settings),
              (FftBasedJaeger2007(), settings)]

reconstructions = reconstruct_ipasc_hdf5(PATH, algorithms)

initial_pressure = (initial_pressure - np.mean(initial_pressure)) / np.std(initial_pressure)
for recon_idx, recon in enumerate(reconstructions):
    reconstructions[recon_idx] = (recon - np.mean(recon)) / np.std(recon)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2,
                                                                     layout="constrained")
ax1.imshow(initial_pressure.T[Y_LOWER:Y_UPPER])
ax2.imshow(np.squeeze(pa_data.binary_time_series_data).T[:1024], aspect=0.1)
ax3.imshow(np.squeeze(reconstructions[0]).T[Y_LOWER:Y_UPPER])
ax5.imshow(np.squeeze(reconstructions[1]).T[Y_LOWER:Y_UPPER])
ax7.imshow(np.squeeze(reconstructions[2]).T[Y_LOWER:Y_UPPER])

ax1.axis('off')
ax3.axis('off')
ax5.axis('off')
ax7.axis('off')
plt.show()
plt.close()
