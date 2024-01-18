import pacfish as pf
import numpy as np
import matplotlib.pyplot as plt
import simpa as sp
from image_reconstruction.reconstruction_algorithms import BackProjection, DelayMultiplyAndSumAlgorithm, FftBasedJaeger2007
from image_reconstruction.batch_reconstruction import reconstruct_ipasc_hdf5
from pw24.utils import create_image_view, calc_results
import os
import mat73
import imageio as iio
from scipy.ndimage import zoom
import patato as pat
from patato.io.ipasc.read_ipasc import IPASCInterface


FILENAME = "3"
BASE_PATH = fr"D:\image_recon_project\finger_data\{FILENAME}/"
PATH_SIM = fr"{BASE_PATH}\Sims/1-PA_ipasc.hdf5"
EXP_DATA_PATH = fr"{BASE_PATH}\exp_ipasc.hdf5"
REF_RECO = fr"{BASE_PATH}\PA images/1-PA.png"
SPACING = 0.0390625 * 2 / 1000
Y_UPPER = 250
Y_LOWER = 50

if not os.path.exists(EXP_DATA_PATH):
    mat_file = mat73.loadmat(fr"D:\image_recon_project\finger_data\{FILENAME}\Rf_PAUSData_{FILENAME}.mat")
    exp_data = mat_file["DataAfterAvePA"][:, :, 0]
    exp_data = np.swapaxes(exp_data, 0, 1)
    exp_data = exp_data.reshape((128, 1024, 1, 1))
    pa_data_exp = pf.load_data(PATH_SIM)
    pa_data_exp.binary_time_series_data = exp_data
    pf.write_data(EXP_DATA_PATH, pa_data_exp)

pa_data_exp = pf.load_data(EXP_DATA_PATH)
pa_data = pf.load_data(PATH_SIM)
ref_reco = iio.imread(REF_RECO)
ref_reco = (ref_reco[:, :, 0] * 0.2126 +
            ref_reco[:, :, 1] * 0.7152 +
            ref_reco[:, :, 2] * 0.0722)

initial_pressure = sp.load_data_field(PATH_SIM.replace("_ipasc", ""), sp.Tags.DATA_FIELD_INITIAL_PRESSURE,
                                      800).astype(float)

segmentation = sp.load_data_field(PATH_SIM.replace("_ipasc", ""),
                                  sp.Tags.DATA_FIELD_SEGMENTATION,
                                  800)

settings = {
            "spacing_m": SPACING,
            "speed_of_sound_m_s": 1540,
            "lowcut": 5e5,
            "highcut": 7e6,
            "order": 3,
            "non_negativity_method": "hilbert",
            "p_factor": 0,
            "p_SCF": 0,
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

reconstructions = reconstruct_ipasc_hdf5(PATH_SIM, algorithms)
exp_reco = reconstruct_ipasc_hdf5(EXP_DATA_PATH, algorithms)

for reco in exp_reco:
    reconstructions.append(np.flip(reco, axis=0).copy())

initial_pressure = initial_pressure[12:-12, 12:-12]
initial_pressure = (initial_pressure - np.mean(initial_pressure)) / np.std(initial_pressure)

for recon_idx, recon in enumerate(reconstructions):
    reconstructions[recon_idx] = (recon - np.nanmean(recon)) / np.nanstd(recon)
    print(np.nanmean(reconstructions[recon_idx]))

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
      (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4,
                                                                        layout="constrained",
                                                                        figsize=(10, 6))
ax1.imshow(initial_pressure.T[Y_LOWER:Y_UPPER])
ax3.imshow(ref_reco[Y_LOWER+25:Y_UPPER+50])
ax2.axis("off")
ax2.set_title("Sim Time Series Data")
ax2.imshow(np.squeeze(pa_data.binary_time_series_data).T[100:600], aspect=0.11)
ax4.axis("off")
ax4.set_title("Exp Time Series Data")
ax4.imshow(np.flip(np.squeeze(pa_data_exp.binary_time_series_data), axis=0).T[100:600], aspect=0.11)


def show(axis, data):
    axis.imshow(data, vmin=np.min(data), vmax=np.percentile(data, 99))


show(ax5, np.squeeze(reconstructions[0]).T[Y_LOWER:Y_UPPER])
show(ax9, np.squeeze(reconstructions[1]).T[Y_LOWER:Y_UPPER])
show(ax13, np.squeeze(reconstructions[2]).T[Y_LOWER:Y_UPPER])
show(ax7, np.squeeze(reconstructions[3]).T[Y_LOWER:Y_UPPER])
show(ax11, np.squeeze(reconstructions[4]).T[Y_LOWER:Y_UPPER])
show(ax15, np.squeeze(reconstructions[5]).T[Y_LOWER:Y_UPPER])

results = list()
results.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[0]).T[Y_LOWER:Y_UPPER]))
results.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[1]).T[Y_LOWER:Y_UPPER]))
results.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[2]).T[Y_LOWER:Y_UPPER]))
results = np.asarray(results)

results_exp = list()
results_exp.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[3]).T[Y_LOWER:Y_UPPER]))
results_exp.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[4]).T[Y_LOWER:Y_UPPER]))
results_exp.append(calc_results(initial_pressure.T[Y_LOWER:Y_UPPER], np.squeeze(reconstructions[5]).T[Y_LOWER:Y_UPPER]))
results_exp = np.asarray(results_exp)

ax6.set_title("Quality Assessment")
create_image_view(ax6, results[0], results)
create_image_view(ax10, results[1], results)
create_image_view(ax14, results[2], results)

ax8.set_title("Quality Assessment")
create_image_view(ax8, results_exp[0], results_exp)
create_image_view(ax12, results_exp[1], results_exp)
create_image_view(ax16, results_exp[2], results_exp)

ax1.set_title("Reference p$_0$")
ax1.axis('off')
ax5.set_title("Backprojection")
ax5.axis('off')
ax9.set_title("DMAS")
ax9.axis('off')
ax13.set_title("FFT-based")
ax13.axis('off')

ax3.set_title("CYBERDYNE Recon")
ax3.axis('off')
ax7.set_title("Backprojection")
ax7.axis('off')
ax11.set_title("DMAS")
ax11.axis('off')
ax15.set_title("FFT-based")
ax15.axis('off')
plt.savefig(f"finger_{FILENAME}.png", dpi=300)

plt.close()
