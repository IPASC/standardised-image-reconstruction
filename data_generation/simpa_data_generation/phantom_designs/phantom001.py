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
from data_generation.simpa_data_generation.utils.custom_tissue import constant


def phantom001(dim_x_mm, dim_y_mm, dim_z_mm):
    background_dictionary = sp.Settings()
    # Near-zero scattering and absorption in the background
    background_dictionary[Tags.MOLECULE_COMPOSITION] = constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    tissue_dict["vessel_1"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[dim_x_mm / 2, 0, dim_z_mm / 2],
        tube_end_mm=[dim_x_mm / 2, dim_y_mm, dim_z_mm / 2],
        molecular_composition=constant(1, 1, 0.9, sp.SegmentationClasses.BLOOD),
        radius_mm=dim_x_mm/10, priority=3, consider_partial_volume=False,
        adhere_to_deformation=False
    )

    return {Tags.STRUCTURES: tissue_dict}