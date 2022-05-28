% Script compute the output of a linear array transducer detecting a
% photoacoustically generated wavefield
%
% SPDX-FileCopyrightText: 2021 International Photoacoustic Standardisation Consortium (IPASC)
% SPDX-FileCopyrightText: 2022 Ben Cox
% SPDX-FileCopyrightText: 2022 Janek Grohl
% SPDX-License-Identifier: MIT
%
% author: Ben Cox, Janek Grohl
% date: 22nd March 2022
% last update: 6th April 2022

clearvars

% =========================================================================
% DEFINE OR LOAD PHOTOACOUSTIC SOURCE
% =========================================================================

% Size of the perfectly matched layer around the domain
PML_size = 15;

% The initial pressure for the study should be a 3D array of these dimensions
%
% Nx = 1024 - 2 * PML = 994
% Ny = 512  - 2 * PML = 482
% Nz = 1024 - 2 * PML = 994
%
% or these, if using the periodicity in k-Wave to generate an infinitely
% long object, eg a cylinder
%
% Nx = 1024 - 2 * PML = 994
% Ny = 512
% Nz = 1024 - 2 * PML = 994


% simple test case for running locally
aNx = 98;
aNy = 64;
aNz = 98;

disc = makeDisc(aNx, aNz, aNx/2, aNz/2, aNx/10);    
initial_pressure = permute(repmat(disc, 1, 1, aNy), [1 3 2]);



% =========================================================================
% SET SIMULATION SETTINGS
% =========================================================================

% Set which code/hardware will be used for the simulations
% 1 - matlab code on CPU
% 2 - C++ code on CPU
% 3 - CUDA code on GPU
computational_model = 3;

% Export the simulated time series in the IPASC hdf5 file format
export_ipasc = true;

% Turn off the PML in k-Wave in the y-direction in order to use the
% periodicity built into k-Wave to make phantom effectively infinite in the
% y-direction.  
infinite_phantom = true;


% =========================================================================
% RUN SIMULATION & SAVE RESULTS
% =========================================================================

% Run the simulation function
time_series_data = ipasc_linear_array_simulation(initial_pressure, computational_model, export_ipasc, infinite_phantom, PML_size);

% Save the result as a mat file (for now - IPASC data format export to come)
save('time_series_data.mat','time_series_data')

% Visualise the simulation result
figure
imagesc(time_series_data)
colorbar
xlabel('time [s]')
ylabel('element number')
title('time series recorded on the linear array')

