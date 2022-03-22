% Script compute the output of a linear array transducer detecting a
% photoacoustically generated wavefield
%
% author: Ben Cox, Janek Grohl
% date: 2nd March 2022
% last update: 22nd March 2022

function [time_series_data] = ipasc_linear_array_simulation( ...
    initial_pressure, export_ipasc,...
    computational_model, y_dim, PML_size)
    arguments
        initial_pressure double
        export_ipasc = true
        computational_model int16 = 3
        y_dim = 512
        PML_size = 15
    end
    
    source.p0 = initial_pressure;
    
    % =========================================================================
    % DEFINE SIMULATION GRID & OTHER PARAMETERS
    % =========================================================================
    
    % select which k-Wave code to run
    % 1: MATLAB CPU code
    % 2: C++ code
    % 3: GPU code
    model = computational_model;
    
    % define the size of the grid
    Nx = 2 * y_dim - 2 * PML_size;
    Ny = y_dim - 2 * PML_size;
    Nz = 2 * y_dim - 2 * PML_size;
    
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
    
    % simulation parameters
    % Turn off PML in the y-direction to simulate an infinite cylinder.
    input_args = {'PMLSize', PML_size, 'PMLInside', false, 'PMLAlpha', [2, 0, 2]...
         'PlotPML', false, 'PlotSim', true, 'Smooth', false, 'DataCast', 'single'};        
     
    % run the relevant model
    switch model
        case 1
            sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});
        case 2
            sensor_data = kspaceFirstOrder3DC(kgrid, medium, source, sensor, input_args{:});
        case 3
            sensor_data = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    end
    
    
    % combine data to give one time series per array element
    combined_sensor_data = sensor_array.combineSensorData(kgrid, sensor_data);    
    time_series_data = combined_sensor_data;

    if export_ipasc
        disp("Exporting to the IPASC data format...")
        kwave_adapter = pacfish.kwave_adapter(sensor_array, time_series_data, medium, kgrid, ...
            [-(N_elements/2 + 0.5)*pitch; (N_elements/2 + 0.5)*pitch; 0; 0; -Nz*dx / 2; Nz*dx / 2]);
        pa_data = kwave_adapter.get_pa_data();
        pacfish.write_data("time_series_data_ipasc.hdf5", pa_data, 1)
    end
    
end
