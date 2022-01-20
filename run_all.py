"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms.test_baseline_delay_and_sum import TestDelayAndSum
from tests.reconstruction_algorithms.test_fftbased_jaeger import TestFFTbasedJaeger
import matplotlib.pyplot as plt

# #####################################################################
# TEST IMAGES DOCUMENTATION
# All images are distributed via the MIT license
# #####################################################################
# #####################################################################
#
# IMAGE_IDX = 0
# Simulated image of two tubular structured underneath a horizontal layer
# provided by Janek Gröhl. SOS=1540
#
# #####################################################################
#
# IMAGE_IDX = 1
# Simulated image of point sources in a homogeneous medium provided by
# Janek Gröhl. SOS=1540
#
# #####################################################################
#
IMAGE_IDX = 2
# Experimental image provided by Manojit Pramanik. It is a point absorber
# in a homogeneous medium. SOS=1480
#
# #####################################################################
#
# IMAGE_IDX = 3
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

SPEED_OF_SOUND = 1480

out = TestDelayAndSum()
out.p_factor = 1
out.fnumber = 0
out.p_SCF = 0
out.speed_of_sound_m_s = SPEED_OF_SOUND
out.lowcut = 5e4
out.highcut = 1e7

out.envelope = True
out.envelope_type = "log"
result1 = out.back_project(IMAGE_IDX, visualise=False)

out.fnumber = 1.
result2 = out.back_project(IMAGE_IDX, visualise=False)

out.p_factor = 1.5
result3 = out.back_project(IMAGE_IDX, visualise=False)
out.fnumber = 0
out.p_factor = 1
out.p_SCF = 1
result4 = out.back_project(IMAGE_IDX, visualise=False)

out.p_SCF = 0
out.p_PCF = 1
result5 = out.back_project(IMAGE_IDX, visualise=False)

out.p_SCF = 2
out.p_PCF = 1
result6 = out.back_project(IMAGE_IDX, visualise=False)

out = TestFFTbasedJaeger()
out.envelope = True
out.envelope_type = "log"
out.speed_of_sound_m_s = SPEED_OF_SOUND
out.time_delay = 0
out.zero_padding_transducer_dimension = 1
out.zero_padding_time_dimension = 1
out.coefficientT = 5
result7 = out.fftbasedJaeger(IMAGE_IDX, visualise=False)

plt.figure(figsize=(12, 8))
plt.subplot(2, 4, 1)
plt.title("BP + Envelope DAS")
plt.imshow(result1[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(2, 4, 2)
plt.title("DAS fnumber")
plt.imshow(result2[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(2, 4, 3)
plt.title("pDAS + fnumber")
plt.imshow(result3[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(2, 4, 4)
plt.title("SCF-DAS + fnumber")
plt.imshow(result4[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(2, 4, 5)
plt.title("PCF-DAS + fnumber")
plt.imshow(result5[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(2, 4, 6)
plt.title("CSF-PCF-DAS + fnumber")
plt.imshow(result6[:, 0, :, 0, 0].T)
plt.colorbar()
plt.subplot(2, 4, 7)
plt.title("FFT-based")
plt.imshow(result7[:, 0, :, 0, 0].T)
plt.colorbar()

plt.tight_layout()
plt.show()

# Only uncomment the following two lines if a new reference reconstruction should be saved!
# import numpy as np
# np.savez("NUMBER_reference.npz",
#          reconstruction=result1[:, 0, :, 0, 0])
