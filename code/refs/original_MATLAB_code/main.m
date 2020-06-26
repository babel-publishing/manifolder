% MATLAB code implementation of the toy example from:
% R. Talmon and R. R. Coifman, "Empirical intrinsic geometry 
% for nonlinear modeling and time series filtering", 
% in PNAS Vol. 110 No. 31 pp.12535?12540.
% ***************************************************************@
% This implementation generates the underlying diffusion processes
% and the corresponding measurements under the Poisson modality and 
% recovers the underlying processes using the proposed EIG method.
% Author: Ronen Talmon.
% Created:  1/6/13.
% ***************************************************************@
% Altered for AVLab Time Series Data by Alex Cloninger, Srinjoy Das

% PLEASE NOTE - the code is included in the repository as a reference,
% and is not officially part of the project, and is not covered
% under the same license (MIT)

%% Configuration
% mat file contains:
% - Dim: dimension to reduce pseudoinverse of covariance matrices to. 
% - z: original time series with each time point as a column
% - x_interp: useful statistics to similarly compress for comparison
%           purposes
clear all; close all; clc;

tic

disp('Load data ...')
load solar_wind_data.mat
N = size(z,1);

% disp('NOTE, using simplified data')
% z = csvread('../../data/simple_data.csv');

%whos
%z(:, [1:3])
%% Modeling

%% Compute histograms
disp('Compute histograms ...')
histograms_overlap;

%% Compute local covariances
disp('Compute local covariances ...')
covariances;

%% Intrinsic modeling
disp('Intrinsic modeling ...')
embedding;

toc % toc here, since clustering does not run through - about 37 seconds

%% Clustering of intrinsic model
disp('Clustering of intrinsic model')
clustering;


