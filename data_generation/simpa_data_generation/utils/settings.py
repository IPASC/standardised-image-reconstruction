from simpa import Settings, Tags
import simpa as sp
import uuid


def generate_base_settings(path_manager: sp.PathManager,
                           speed_of_sound=1540,
                           volume_name=str(uuid.uuid4()),
                           wavelength=800,
                           random_seed=1337) -> Settings:
    """
    This function will return the base settings which are already optimised for
    the IPASC standardised image reconstruction project needs.

    Settings for the respective volume generation module will need to be called
    separately.

    :return: dict
        A `simpa.Settings` instance with optimised entries for the IPASC image
        reconstruction project.
    """

    settings = Settings()
    settings[Tags.SIMULATION_PATH] = path_manager.get_hdf5_file_save_path()
    settings[Tags.VOLUME_NAME] = volume_name
    settings[Tags.DIM_VOLUME_X_MM] = 38.828125
    settings[Tags.DIM_VOLUME_Y_MM] = 18.828125
    settings[Tags.DIM_VOLUME_Z_MM] = 38.828125
    settings[Tags.SPACING_MM] = 0.0390625 * 4
    settings[Tags.DATA_FIELD_SPEED_OF_SOUND] = speed_of_sound
    settings[Tags.WAVELENGTHS] = [wavelength]
    settings[Tags.RANDOM_SEED] = random_seed

    settings.set_acoustic_settings({
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
    })

    settings.set_reconstruction_settings({
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.DATA_FIELD_SPEED_OF_SOUND: speed_of_sound,
        Tags.DATA_FIELD_ALPHA_COEFF: 0.01,
        Tags.DATA_FIELD_DENSITY: 1000,
        Tags.SPACING_MM: settings[Tags.SPACING_MM]
    })

    return settings