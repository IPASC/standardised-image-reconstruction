# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-FileCopyrightText: 2022 Mengjie Shi
# SPDX-License-Identifier: MIT

"""
Phantom_random_vessels:

This is a script of generating random spatially distributed circular tubular targets as blood vessels at a skin layer.
It contains an epidermis layer (1 mm) on top of a dermis layer (3 mm), followed by a hypodermis background (muscle) contains 1-5 blood vessels.

"""

from simpa import Tags
import simpa as sp
from data_generation.simpa_data_generation.utils.custom_tissue import constant
from random import choice

def phantom_random_vessels(dim_x_mm, dim_y_mm, dim_z_mm):

    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    epidermis_dictionary = sp.Settings()
    epidermis_dictionary[Tags.PRIORITY] = 8
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 5]
    epidermis_dictionary[Tags.STRUCTURE_END_MM]= [0, 0, 6]
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.epidermis()
    epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    dermis_dictionary = sp.Settings()
    dermis_dictionary[Tags.PRIORITY] = 1
    dermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 6]
    dermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 9]
    dermis_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.dermis()
    dermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    dermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    dermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    muscle_dictionary = sp.Settings()
    muscle_dictionary[Tags.PRIORITY] = 1
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 10]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle()
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = True
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["dermis"] = dermis_dictionary
    tissue_dict["muscle"] = muscle_dictionary

    # random create 1-5 vessels in the simulation (dermis) space
    num_vessels = choice(range(1, 6, 1))
    for j in range(0, num_vessels):
        vessel_dictionary = sp.Settings()
        vessel_dictionary[Tags.PRIORITY] = 3
        start_pos_x = choice(range(10, 20, 2))
        start_pos_y = choice(range(10, 20, 2))
        vessel_dictionary[Tags.STRUCTURE_START_MM] = [start_pos_x, 0, start_pos_y]
        vessel_dictionary[Tags.STRUCTURE_END_MM] = [start_pos_x, dim_y_mm, start_pos_y]
        vessel_dictionary[Tags.STRUCTURE_RADIUS_MM] = 0.5*choice(range(1, 5, 1))
        vessel_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood()
        vessel_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = True
        vessel_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
        vessel_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE
        tissue_dict["vessel_%d" % j] = vessel_dictionary

    return {Tags.STRUCTURES: tissue_dict}