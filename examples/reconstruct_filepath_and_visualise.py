"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-License-Identifier: MIT
"""

from image_reconstruction.reconstruction_algorithms import BackProjection
import matplotlib.pyplot as plt
from pacfish import load_data

# TODO Add path to IPASC HDF5 file here.
name = "Data_0729_1825_foot_try1_80mm_0_4mm_PA_Rcv.mat_68"
#FILE_PATH = f"../10ZOB6_Y24gexHgQwMHWUMyJl6GAYppAx_ipasc.hdf5"
FILE_PATH = rf"C:\Users\jgroe\Downloads\POSTECH\{name}.hdf5"

params = {
    "spacing_m": 0.0002,
    "speed_of_sound_m_s": 1500,
    "lowcut": None,
    "highcut": 5e6,
    "order": 3,
    "envelope": True,
    "p_factor": 1,
    "p_SCF": 0,
    "p_PCF": 0,
    "fnumber": 0,
    "envelope_type": "hilbert",
    "time_interpolation_factor": 3,
    "detector_interpolation_factor": 3
         }

algo = BackProjection()
reco = algo.reconstruct_time_series_data(FILE_PATH, **params)

ipasc_data = load_data(FILE_PATH)
field_of_view = ipasc_data.get_field_of_view()

plt.figure(figsize=(4.75, 4))
plt.title("Reconstruction Result")
plt.imshow(reco[:, 0, :, 0, 0].T, extent=[field_of_view[0]*100, field_of_view[1]*100,
                                          field_of_view[4]*100, field_of_view[5]*100])
plt.xlabel("Image width [cm]")
plt.ylabel("Image height [cm]")
cb = plt.colorbar()
cb.set_label("Signal Amplitude [a.u.]")
plt.tight_layout()
plt.savefig(f"{name}.png", dpi=300)
plt.show()


