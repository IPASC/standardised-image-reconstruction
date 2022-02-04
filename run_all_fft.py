"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Jenni Poimala
SPDX-FileCopyrightText: 2021 Andreas Hauptmann
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms.test_baseline_fft_reconstruction import TestFFTRecon
import matplotlib.pyplot as plt

# #####################################################################
# TEST IMAGES DOCUMENTATION
# All images are distributed via the MIT license
# #####################################################################
# #####################################################################
#
#IMAGE_IDX = 0
# Simulated image of two tubular structured underneath a horizontal layer
# provided by Janek Gröhl. SOS=1540
#
# #####################################################################
#
#IMAGE_IDX = 1
# Simulated image of point sources in a homogeneous medium provided by
# Janek Gröhl. SOS=1540
#
# #####################################################################
#
#IMAGE_IDX = 2
# Experimental image provided by Manojit Pramanik. It is a point absorber
# in a homogeneous medium. SOS=1480
#
# #####################################################################
#
IMAGE_IDX = 3
# Simulated image of point sources in a homogeneous medium provided by
# François Varray. 10 point absorbers are located in a homogeneous medium
# at depths between 10 and 40 mm. With increasing depth, they are
# also positioned laterally between 0 and 30 mm. SOS=1540
#
# #####################################################################
#
# IMAGE_IDX = 4
# Experimental measurement of a point source in a homogeneous medium.
# Measurement is provided by Mengjie Shi. Apparent SOS: 1380
#
# #####################################################################
#
# IMAGE_IDX = 5
# Experimental measurement of a point source in a homogeneous medium.
# Measurement is provided by Mengjie Shi. Apparent SOS: 1380
#
# #####################################################################

SPEED_OF_SOUND = 1540

out = TestFFTRecon()
out.speed_of_sound_m_s = SPEED_OF_SOUND
out.lowcut = 5e4
out.highcut = 1e7

out.envelope = False
out.envelope_type = "log"
result1 = out.fft_recon(IMAGE_IDX, visualise=False)



plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.title("FFT based reconstruction")
plt.imshow(result1[:, 0, :, 0, 0].T)
plt.colorbar()


plt.tight_layout()
plt.show()

# Only uncomment the following two lines if a new reference reconstruction should be saved!
# import numpy as np
# np.savez("NUMBER_reference.npz",
#          reconstruction=result1[:, 0, :, 0, 0])
