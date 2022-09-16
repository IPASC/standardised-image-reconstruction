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


def phantom002(dim_x_mm, dim_y_mm, dim_z_mm):
    RADIUS = 0.3
    background_dictionary = sp.Settings()
    # Near-zero scattering and absorption in the background
    background_dictionary[Tags.MOLECULE_COMPOSITION] = constant(1e-10, 1e-10, 1.0)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary

    tissue_dict["vessel_1"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[5, 0, 5],
        tube_end_mm=[5, dim_y_mm, 5],
        molecular_composition=constant(1, 1, 0.9, sp.SegmentationClasses.BLOOD),
        radius_mm=RADIUS, priority=3, consider_partial_volume=False,
        adhere_to_deformation=False
    )

    tissue_dict["vessel_2"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[10, 0, 10],
        tube_end_mm=[10, dim_y_mm, 10],
        molecular_composition=constant(1, 1, 0.9, sp.SegmentationClasses.BLOOD),
        radius_mm=RADIUS, priority=3, consider_partial_volume=False,
        adhere_to_deformation=False
    )

    tissue_dict["vessel_3"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[15, 0, 15],
        tube_end_mm=[15, dim_y_mm, 15],
        molecular_composition=constant(1, 1, 0.9, sp.SegmentationClasses.BLOOD),
        radius_mm=RADIUS, priority=3, consider_partial_volume=False,
        adhere_to_deformation=False
    )

    tissue_dict["vessel_4"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[20, 0, 20],
        tube_end_mm=[20, dim_y_mm, 20],
        molecular_composition=constant(1, 1, 0.9, sp.SegmentationClasses.BLOOD),
        radius_mm=RADIUS, priority=3, consider_partial_volume=False,
        adhere_to_deformation=False
    )

    tissue_dict["vessel_5"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[25, 0, 25],
        tube_end_mm=[25, dim_y_mm, 25],
        molecular_composition=constant(1, 1, 0.9, sp.SegmentationClasses.BLOOD),
        radius_mm=RADIUS, priority=3, consider_partial_volume=False,
        adhere_to_deformation=False
    )

    tissue_dict["vessel_6"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[30, 0, 30],
        tube_end_mm=[30, dim_y_mm, 30],
        molecular_composition=constant(1, 1, 0.9, sp.SegmentationClasses.BLOOD),
        radius_mm=RADIUS, priority=3, consider_partial_volume=False,
        adhere_to_deformation=False
    )

    tissue_dict["vessel_7"] = sp.define_circular_tubular_structure_settings(
        tube_start_mm=[35, 0, 35],
        tube_end_mm=[35, dim_y_mm, 35],
        molecular_composition=constant(1, 1, 0.9, sp.SegmentationClasses.BLOOD),
        radius_mm=RADIUS, priority=3, consider_partial_volume=False,
        adhere_to_deformation=False
    )

    return {Tags.STRUCTURES: tissue_dict}