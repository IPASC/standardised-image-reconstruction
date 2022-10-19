"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Andreas Hauptmann
SPDX-FileCopyrightText: 2022 Jenni Poimala
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms.test_baseline_delay_and_sum import TestDelayAndSum
from tests.reconstruction_algorithms.test_delay_multiply_and_sum import TestDelayMultiplyAndSum
from tests.reconstruction_algorithms.test_fftbased_jaeger import TestFFTbasedJaeger
from tests.reconstruction_algorithms.test_baseline_fft_reconstruction import TestFFTbasedHauptmann

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
# IMAGE_IDX = 2
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
#
IMAGE_IDX = 6
# Experimental measurement of a foot.
# Measurement is provided by Minsik Sung. Apparent SOS: 1500
#
# #####################################################################

SPEED_OF_SOUND = 1500
ENVELOPE_TYPE = "log"  # One of "log", "hilbert", "abs", "zero", "hilbert_squared", "log_squared"
LOWCUT = None  # 5e4
HIGHCUT = None  # 1e7

out = TestDelayAndSum()
out.p_factor = 1
out.fnumber = 0
out.p_SCF = 0
out.speed_of_sound_m_s = SPEED_OF_SOUND
out.lowcut = LOWCUT
out.highcut = HIGHCUT
out.envelope = True
out.envelope_type = ENVELOPE_TYPE

result1 = out.back_project(IMAGE_IDX, visualise=False)

out.fnumber = 1.0
result2 = out.back_project(IMAGE_IDX, visualise=False)

out.fnumber = 0
out.p_factor = 1.5
result3 = out.back_project(IMAGE_IDX, visualise=False)

out.fnumber = 0
out.p_factor = 1
out.p_SCF = 1
result4 = out.back_project(IMAGE_IDX, visualise=False)

out.p_SCF = 0
out.p_PCF = 1
result5 = out.back_project(IMAGE_IDX, visualise=False)

out = TestFFTbasedJaeger()
out.envelope = True
out.envelope_type = ENVELOPE_TYPE
out.speed_of_sound_m_s = SPEED_OF_SOUND
out.time_delay = 0
out.zero_padding_transducer_dimension = 1
out.zero_padding_time_dimension = 1
out.coefficientT = 5
out.lowcut = LOWCUT
out.highcut = HIGHCUT
result6 = out.fftbasedJaeger(IMAGE_IDX, visualise=False)

out = TestFFTbasedHauptmann()
out.speed_of_sound_m_s = SPEED_OF_SOUND
out.lowcut = LOWCUT
out.highcut = HIGHCUT
out.envelope = True
out.envelope_type = ENVELOPE_TYPE
result7 = out.fft_recon(IMAGE_IDX, visualise=False)

# out = TestDelayMultiplyAndSum()
# out.speed_of_sound_m_s = SPEED_OF_SOUND
# out.lowcut = LOWCUT
# out.highcut = HIGHCUT
# out.envelope = True
# out.envelope_type = ENVELOPE_TYPE
# out.fnumber = 0
# out.signed_dmas = False
# result8 = out.back_project(IMAGE_IDX, visualise=False)
#
# out = TestDelayMultiplyAndSum()
# out.speed_of_sound_m_s = SPEED_OF_SOUND
# out.lowcut = LOWCUT
# out.highcut = HIGHCUT
# out.envelope = True
# out.envelope_type = ENVELOPE_TYPE
# out.fnumber = 1
# out.signed_dmas = False
# result9 = out.back_project(IMAGE_IDX, visualise=False)
#
# out = TestDelayMultiplyAndSum()
# out.speed_of_sound_m_s = SPEED_OF_SOUND
# out.lowcut = LOWCUT
# out.highcut = HIGHCUT
# out.envelope = True
# out.envelope_type = ENVELOPE_TYPE
# out.fnumber = 0
# out.signed_dmas = True
# result10 = out.back_project(IMAGE_IDX, visualise=False)
#
# out = TestDelayMultiplyAndSum()
# out.speed_of_sound_m_s = SPEED_OF_SOUND
# out.lowcut = LOWCUT
# out.highcut = HIGHCUT
# out.envelope = True
# out.envelope_type = ENVELOPE_TYPE
# out.fnumber = 1
# out.signed_dmas = True
# result11 = out.back_project(IMAGE_IDX, visualise=False)


vmin = None
vmax = None

if ENVELOPE_TYPE == "log" or ENVELOPE_TYPE == "log_squared":
    vmin = -40
    vmax = 0

plt.figure(figsize=(12, 9))
plt.subplot(3, 4, 1)
plt.title("DAS")
plt.imshow(result1[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(3, 4, 2)
plt.title("DAS + fnumber")
plt.imshow(result2[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(3, 4, 3)
plt.title("DAS + p-factor")
plt.imshow(result3[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(3, 4, 4)
plt.title("DAS + SCF")
plt.imshow(result4[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(3, 4, 5)
plt.title("DAS + PCF")
plt.imshow(result5[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(3, 4, 6)
plt.title("FFT-based (Jaeger)")
plt.imshow(result6[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(3, 4, 7)
plt.title("FFT-based (Hauptmann)")
plt.imshow(result7[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
# plt.subplot(3, 4, 8)
# plt.title("DMAS")
# plt.imshow(result8[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.subplot(3, 4, 9)
# plt.title("DMAS + fnumber")
# plt.imshow(result9[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.subplot(3, 4, 10)
# plt.title("sDMAS")
# plt.imshow(result10[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.subplot(3, 4, 11)
# plt.title("sDMAS + fnumber")
# plt.imshow(result11[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
# plt.colorbar()

plt.tight_layout()
plt.show()
