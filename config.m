% 00_config.m
% Configuration file for the pipeline

clear; clc;
rng(0);

% --- Paths
rootFolder = pwd;     % assume current folder
dataFolder = fullfile(rootFolder,'data');

% --- Windowing params
params.window_len = 128;  % samples per window (~4s at ~32Hz)
params.hop = 32;          % 75% overlap
params.minSamplesPerFile = 500;

% --- Normalization
params.normalize_method = 'minmax'; % 'minmax' or 'zscore'

% --- Models params
params.knn_k_candidates = [1 3 5 7 9];
params.ffnn_layers = [128 64 32];
params.ffnn_epochs = 400;
params.ffnn_maxfail = 10;

% --- Save/flags
params.saveMat = true;

% Save config
save('config.mat','dataFolder','params');
fprintf("Config saved (config.mat). Data folder: %s\n", dataFolder);
