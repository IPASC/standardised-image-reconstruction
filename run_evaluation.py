"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2021 Andreas Hauptmann
SPDX-FileCopyrightText: 2022 Jenni Poimala
SPDX-FileCopyrightText: 2021 Janek Gröhl
SPDX-FileCopyrightText: 2021 François Varray
SPDX-License-Identifier: MIT
"""

from tests.reconstruction_algorithms.test_baseline_delay_and_sum import TestDelayAndSum

import matplotlib.pyplot as plt
import numpy as np
from quality_assessment.measures.no_reference.gsnr import GeneralisedSignalToNoiseRatio

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
ENVELOPE_TYPE = "log"  # One of "log", "hilbert", "abs", "zero", "hilbert_squared", "log_squared"
LOWCUT = None  # 5e4
HIGHCUT = None  # 1e7

gsnr = GeneralisedSignalToNoiseRatio()

SIGNAL_ROI = np.zeros((384, 576))
SIGNAL_ROI[139:159, 330:350] = 1

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

vmin = None
vmax = None

if ENVELOPE_TYPE == "log" or ENVELOPE_TYPE == "log_squared":
    vmin = -40
    vmax = 0

plt.figure(figsize=(12, 9))
plt.subplot(2, 3, 1)
gsnr_value1 = gsnr.compute_measure(result1[:, 0, :, 0, 0], SIGNAL_ROI == 1, SIGNAL_ROI == 0)
plt.title(f"DAS ({gsnr_value1})")
plt.imshow(result1[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 3, 2)
gsnr_value2 = gsnr.compute_measure(result2[:, 0, :, 0, 0], SIGNAL_ROI == 1, SIGNAL_ROI == 0)
plt.title(f"DAS + fnumber ({gsnr_value2})")
plt.imshow(result2[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 3, 3)
gsnr_value3 = gsnr.compute_measure(result3[:, 0, :, 0, 0], SIGNAL_ROI == 1, SIGNAL_ROI == 0)
plt.title(f"DAS + p-factor ({gsnr_value3})")
plt.imshow(result3[:, 0, :, 0, 0].T, vmin=vmin, vmax=vmax)
plt.colorbar()

result1 = result1[:, 0, :, 0, 0]
result1[SIGNAL_ROI==0] = np.nan
plt.subplot(2, 3, 4)
plt.title(f"DAS ({gsnr_value1})")
plt.imshow(result1.T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 3, 5)
plt.title(f"DAS + fnumber ({gsnr_value2})")
result2 = result2[:, 0, :, 0, 0]
result2[SIGNAL_ROI==0] = np.nan
plt.imshow(result2.T, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 3, 6)
plt.title(f"DAS + p-factor ({gsnr_value3})")
result3 = result3[:, 0, :, 0, 0]
result3[SIGNAL_ROI==0] = np.nan
plt.imshow(result3.T, vmin=vmin, vmax=vmax)
plt.colorbar()

plt.tight_layout()
plt.show()
