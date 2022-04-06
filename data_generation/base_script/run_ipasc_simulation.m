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
% last update: 22nd March 2022

clearvars

% =========================================================================
% DEFINE PHOTOACOUSTIC SOURCE
% =========================================================================
%
% The dimensions have to be:
% Nx = 1024 - 2 * PML = 994
% Ny = 512 - 2 * PML = 482
% Nz = 1024 - 2 * PML = 994
%
% The output has to be a 3D array of these dimensions of type double

aNx = 98;
aNy = 34;
aNz = 98;

disc = makeDisc(aNx, aNz, aNx/2, aNz/2, aNx/10);    
initial_pressure = permute(repmat(disc, 1, 1, aNy), [1 3 2]);

% =========================================================================
% SIMULATION
% =========================================================================

% Run the simulation function
time_series_data = ipasc_linear_array_simulation(initial_pressure, true, 3, 64);

% Save the result as a mat file (for now - IPASC data format export to come)
save('time_series_data.mat','time_series_data')

% Visualise the simulation result
figure
imagesc(time_series_data)
colorbar
xlabel('time [s]')
ylabel('element number')
title('time series recorded on the linear array')
