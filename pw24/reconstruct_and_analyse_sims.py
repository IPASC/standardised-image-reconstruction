import pacfish as pf
import numpy as np
import matplotlib.pyplot as plt
import simpa as sp
from image_reconstruction.reconstruction_algorithms import BackProjection, DelayMultiplyAndSumAlgorithm, FftBasedJaeger2007
from image_reconstruction.batch_reconstruction import reconstruct_ipasc_hdf5
from pw24.utils import create_image_view, calc_results

#PATH = r"D:\image_recon_project\finger_data\4\Sims/1-PA_ipasc.hdf5"
#SPACING = 0.0390625 * 2 / 1000
# BlobPhantom_4712
# StringPhantom_4712
# SkinModel_4713
FILENAME = "SkinModel_4713"
PATH = fr"D:\image_recon_project\in_silico_data\{FILENAME}_ipasc.hdf5"
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

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2,
                                                                     layout="constrained",
                                                                     gridspec_kw={
                                                                         "width_ratios": [1, 2]
                                                                     })
ax1.imshow(initial_pressure.T[Y_LOWER:Y_UPPER])
ax2.axis("off")
ax2.set_title("Time Series Data")
ax2.imshow(np.squeeze(pa_data.binary_time_series_data).T[50:450], aspect=0.1)
ax3.imshow(np.squeeze(reconstructions[0]).T[Y_LOWER:Y_UPPER])
ax5.imshow(np.squeeze(reconstructions[1]).T[Y_LOWER:Y_UPPER])
ax7.imshow(np.squeeze(reconstructions[2]).T[Y_LOWER:Y_UPPER])

results = []
results.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[0]).T[Y_LOWER:Y_UPPER]))
results.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[1]).T[Y_LOWER:Y_UPPER]))
results.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[2]).T[Y_LOWER:Y_UPPER]))
results = np.asarray(results)

ax4.set_title("Quality Assessment")
create_image_view(ax4, results[0], results)
create_image_view(ax6, results[1], results)
create_image_view(ax8, results[2], results)

ax1.set_title("Reference p$_0$")
ax1.axis('off')
ax3.set_title("Backprojection")
ax3.axis('off')
ax5.set_title("DMAS")
ax5.axis('off')
ax7.set_title("FFT-based")
ax7.axis('off')
plt.savefig(f"{FILENAME}.png", dpi=300)
plt.close()
