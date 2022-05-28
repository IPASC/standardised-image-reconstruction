% Script compute the output of a linear array transducer detecting a
% photoacoustically generated wavefield
%
% SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
% SPDX-FileCopyrightText: 2022 Ben Cox
% SPDX-FileCopyrightText: 2022 Janek Grohl
% SPDX-License-Identifier: MIT
%
% author: Ben Cox, Janek Grohl
% date: 2nd March 2022
% last update: 26th May 2022

function [time_series_data] = ipasc_linear_array_simulation( ...
        load_path, computational_model, export_ipasc, ...
        infinite_phantom, PML_size)

    arguments
        load_path string
        computational_model int16 = 3
        export_ipasc logical = true
        infinite_phantom logical = false
        PML_size {mustBeNumeric} = 15
    end

    data = load(load_path);
    initial_pressure = data.initial_pressure;

    % set the initial pressure
    source.p0 = initial_pressure;

    % =========================================================================
    % DEFINE SIMULATION GRID & OTHER PARAMETERS
    % =========================================================================

    % extract the size of the initial pressure volume
    [Nx,Ny,Nz] = size(initial_pressure);

    % Check the initial pressure is the right size
    if infinite_phantom
        if (Nx~=Nz)||(Ny~=(Nx+2*PML_size)/2)
            warning(['For these simulations, the x and z dimensions should be the same,'...
                ' and the y dimension (not including the PML) should be half the other dimensions (when they include the PML).'])
        end
    else
        if (Nx~=Nz)||((Ny+2*PML_size)~=(Nx+2*PML_size)/2)
            warning(['For these simulations, the x and z dimensions should be the same,'...
                ' and the y dimension should be half the other dimensions (when the PMLs are included).'])
        end
    end

    % define the domain size in the x-direction
    Lx = 40e-3;     % [m]

    % calculate the (isotropic) grid spacing
    dx = Lx/Nx;

    % create the computational grid
    kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);

    % medium properties
    medium.sound_speed = 1540;        % [m/s]

    % set sampling rate to 50Mhz
    dt = 1.0 / double(50000000);
    % Simulate as many time steps as a wave takes to traverse diagonally through the entire tissue
    Nt = round((sqrt(Ny*Ny+Nx*Nx+Nz*Nz)*dx / mean(medium.sound_speed, 'all')) / dt);
    kgrid.setTime(Nt, dt);

    % =========================================================================
    % DEFINE SENSOR ARRAY
    % =========================================================================

    % create empty kWaveArray object
    sensor_array = kWaveArray;

    % define rectangular element size, orientation, and array pitch
    Lx    = 0.2e-3;       % [m]
    Ly    = 8e-3;         % [m]
    theta = [0, 0, 0];    % [deg]
    pitch = 0.3e-3;       % [m]

    % add elements to form the linear array
    N_elements = 128;                                   % number of elements
    y_position  = -Ly/2;                                    % [m]
    z_position  = min(kgrid.z_vec);                    % [m]
    for N_loop = 0:N_elements-1
        x_position  = (N_loop - N_elements/2 + 0.5) * pitch;    % [m]
        sensor_array.addRectElement([x_position, y_position, z_position], Lx, Ly, theta)
    end

    % assign off-grid sensor to source structure for input to kspaceFirstOrder3D
    sensor.mask = sensor_array.getArrayBinaryMask(kgrid);

    % =========================================================================
    % RUN SIMULATION
    % =========================================================================

    % Set the simulation parameters

    if infinite_phantom
        % Turn off PML in the y-direction to simulate an infinite cylinder.
        input_args = {'PMLSize', [PML_size, 0, PML_size], 'PMLInside', false, 'PMLAlpha', [2, 0, 2]...
            'PlotPML', false, 'Smooth', false, 'DataCast', 'single'};
    else
        input_args = {'PMLSize', PML_size, 'PMLInside', false, ...
            'PlotPML', false, 'Smooth', false, 'DataCast', 'single'};
    end

    % run the chosen code; plot the output if running the matlab version
    switch computational_model
        case 1
            input_args = [input_args(:)', {'PlotSim'}, {true}];
            sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
        case 2
            input_args = [input_args(:)', {'PlotSim'}, {false}];
            sensor_data = kspaceFirstOrder3DC(kgrid, medium, source, sensor, input_args{:});
        case 3
            input_args = [input_args(:)', {'PlotSim'}, {false}];
            sensor_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    end


    % combine data to give one time series per array element
    combined_sensor_data = sensor_array.combineSensorData(kgrid, sensor_data);
    time_series_data = combined_sensor_data;

    save_path = strrep(load_path,"_kwave.mat","_ipasc.hdf5");

    if export_ipasc
        disp("Exporting to the IPASC data format...")
        kwave_adapter = pacfish.kwave_adapter(sensor_array, time_series_data, medium, kgrid, ...
            [-(N_elements/2 + 0.5)*pitch; (N_elements/2 + 0.5)*pitch; 0; 0; -Nz*dx / 2; Nz*dx / 2]);
        pa_data = kwave_adapter.get_pa_data();
        pacfish.write_data(save_path, pa_data, 1)
    end

end
