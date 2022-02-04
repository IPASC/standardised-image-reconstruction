"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-License-Identifier: MIT
"""

from image_reconstruction.reconstruction_algorithms import BackProjection
import matplotlib.pyplot as plt

# TODO Add path to IPASC HDF5 file here.
name = "kwave_2Dsim_circular_array_new"
FILE_PATH = f"C:/kWave-PACFISH-export/{name}.hdf5"

params = {
            "spacing_m": 0.0001
         }

algo = BackProjection()
reco = algo.reconstruct_time_series_data(FILE_PATH, **params)

plt.figure(figsize=(4.75, 4))
plt.title("Reconstruction Result")
plt.imshow(reco[:, 0, :, 0, 0].T, extent=[0, 6, 0, 6])
plt.xlabel("Image width [cm]")
plt.ylabel("Image height [cm]")
cb = plt.colorbar()
cb.set_label("Signal Amplitude [a.u.]")
plt.tight_layout()
plt.savefig(f"C:/kWave-PACFISH-export/{name}.png", dpi=300)

