"""
SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
SPDX-FileCopyrightText: 2022 Janek Grohl
SPDX-License-Identifier: MIT
"""

from simpa.utils.libraries.structure_library import CircularTubularStructure
from simpa import Settings, Tags, TISSUE_LIBRARY
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

SPACING = 0.0390625
mock_global_settings = Settings()
mock_global_settings[Tags.DIM_VOLUME_X_MM] = 38.828125
mock_global_settings[Tags.DIM_VOLUME_Y_MM] = 18.828125
mock_global_settings[Tags.DIM_VOLUME_Z_MM] = 38.828125
mock_global_settings[Tags.SPACING_MM] = SPACING
mock_global_settings.set_volume_creation_settings(dict())
START_SLICE = 0
END_SLICE = 20

# ####################################################
# Circular targets
# ####################################################

print("Generating tube 1...")

structure_settings_1 = Settings({
    Tags.STRUCTURE_START_MM: [20, START_SLICE, 20],
    Tags.STRUCTURE_END_MM: [20, END_SLICE, 20],
    Tags.STRUCTURE_RADIUS_MM: 2,
    Tags.CONSIDER_PARTIAL_VOLUME: False,
    Tags.MOLECULE_COMPOSITION: TISSUE_LIBRARY.constant(1, 100, 0.9)
})
struc_1 = CircularTubularStructure(mock_global_settings, structure_settings_1)
tube_1 = struc_1.geometrical_volume

print("Generating tube 1...[Done]")

tube = tube_1

print(np.shape(tube))

plt.imshow(tube[:, 200, :])
plt.savefig("test.png")
plt.close()

scipy.io.savemat('../simpa_geometric_structures.mat', {'array': tube})
