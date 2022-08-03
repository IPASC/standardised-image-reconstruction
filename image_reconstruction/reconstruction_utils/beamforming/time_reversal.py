from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave import kWaveMedium, kWaveGrid
from kwave.ksource import kSource
from kwave.ksensor import kSensor
import numpy as np


def time_reversal_kwave_wrapper(time_series_data,
                                detection_elements,
                                sampling_rate,
                                sos,
                                spacing_m,
                                field_of_view):
    """
    TODO

    :param time_series_data:
    :param detection_elements:
    :param sampling_rate:
    :param sos:
    :param spacing_m:
    :param field_of_view:
    :return:
    """

    DATA_CAST = 'single'

    detector_positions = detection_elements["positions"]
    # calculate Nx, Ny, Nz from field of view and detector positions
    fov_extent = np.asarray([np.abs(field_of_view[0] - field_of_view[1]),
                             np.abs(field_of_view[2] - field_of_view[3]),
                             np.abs(field_of_view[4] - field_of_view[5])])
    detection_elements_extent = np.asarray([np.abs(np.max(detector_positions[:, 0]) - np.min(detector_positions[:, 0])),
                                            np.abs(np.max(detector_positions[:, 1]) - np.min(detector_positions[:, 1])),
                                            np.abs(np.max(detector_positions[:, 2]) - np.min(detector_positions[:, 2]))])

    nx = int(np.max([fov_extent[0], detection_elements_extent[0]]) / spacing_m)
    ny = int(np.max([fov_extent[1], detection_elements_extent[1]]) / spacing_m)
    nz = int(np.max([fov_extent[2], detection_elements_extent[2]]) / spacing_m)

    if nx == 0:
        nx = 1
    if ny == 0:
        ny = 1
    if nz == 0:
        nz = 1

    print(nx, ny, nz)

    # create binary detector mask
    sensor_mask = np.zeros((nx, ny, nz))

    sensor = kSensor()
    sensor.time_reversal_boundary_data = time_series_data
    sensor.mask = sensor_mask

    # define source medium
    source = kSource()

    kgrid = kWaveGrid([nx, ny, nz], [spacing_m, spacing_m, spacing_m])

    dt = 1 / sampling_rate
    Nt = int(np.ceil((np.sqrt((nx*spacing_m)**2 + (ny*spacing_m)**2 + (nz*spacing_m)**2) / sos) / dt))
    print(Nt, dt)
    kgrid.setTime(Nt, dt)

    medium = kWaveMedium(
        sound_speed=sos,
        density=1000,      # TODO make this a parameter
        alpha_coeff=0.75,  # TODO make this a parameter
        alpha_power=1.5    # TODO make this a parameter
    )

    input_args = {
        'PMLInside': False,
        'PMLSize': [32, 32, 32],
        'DataCast': DATA_CAST,
        'DataRecast': True
    }

    reconstruction = kspaceFirstOrder3DC(**{
        'medium': medium,
        'kgrid': kgrid,
        'source': source,
        'sensor': sensor,
        **input_args
    })

    return reconstruction

