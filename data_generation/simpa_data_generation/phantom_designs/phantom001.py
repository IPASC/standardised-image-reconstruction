# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT

"""
Phantom 001:

TODO description of the phantom that is created with this script.

TODO description how the simulation pipeline needs to be altered to use this phantom definition.
"""

from simpa import Tags
import simpa as sp


def phantom001():
    background_dictionary = sp.Settings()
    # Near-zero scattering and absorption in the background
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    return {Tags.STRUCTURES: tissue_dict}