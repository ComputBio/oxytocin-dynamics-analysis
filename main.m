%% Analysis of Oxytocin (OT) Vesicle Dynamics via Machine Learning
% This script:
% 1. Preprocesses experimental trajectories.
% 2. Generates synthetic trajectories (fBM) for training.
% 3. Trains and evaluates a Random Forest (RF) classifier.
% 4. Trains and evaluates a 1D Convolutional Neural Network (CNN).
% 5. Classifies experimental data and evaluates model consensus.

clearvars; close all; clc;

%% 0. Configuration & Parameters
% Define global parameters 
params.numSteps = 80;        % Number of frames per trajectory
params.timeStep = 1;         % Seconds per frame
params.numTrees = 300;       % RF ensemble size
params.numSynthetic = 1e5;   % Size of synthetic training set
params.rngSeed = 1;          % For reproducibility

% Global parameters used by synthetic generation and training
N_synthetic = 1e5;   % Total number of synthetic trajectories
numSteps    = 80;    % Frames per trajectory
timestep    = 1;     % Seconds per frame (Delta T)

% Target proportions for training (must sum to 1)
N_nd  = 0.35;        % 35% Normal Diffusion
N_sub = 0.40;        % 40% Subdiffusion
N_sup = 0.25;        % 25% Superdiffusion

params.numTrees = 300;       % RF ensemble size
params.rngSeed  = 1;         % For reproducibility

% Create directories if they don't exist
if ~exist('results', 'dir'), mkdir('results'); end

%% 1. Data Loading and Preprocessing
fprintf('--- Step 1: Experimental Data Preparation ---\n');

dataPath = fullfile('data', 'prepared_data.mat');
dimPath  = fullfile('data', 'T_dim.mat');

if exist(dataPath, 'file')
    fprintf('Loading prepared experimental data...\n');
    load(dataPath); 
else
    fprintf('Running data preparation pipeline...\n');
    data_preparation; 
end

% Load diameter information (for the size-impact analysis)
if exist(dimPath, 'file')
    load(dimPath);
else
    error('Vesicle diameter data (T_dim.mat) not found.');
end

valid_D_indices = ~isnan(T.D) & T.D > 0;
logD_real = log10(T.D(valid_D_indices));
logD_mean_real = mean(logD_real);
logD_std_real = std(logD_real);

% Summary Statistics
fprintf('Experimental Dataset: %d trajectories\n', height(T));
fprintf('Mean Alpha: %.3f | Mean D: %.3f\n', mean(T.Alpha, 'omitnan'), mean(T.D, 'omitnan'));

%% 2. Synthetic Data Generation
fprintf('\n--- Step 2: Generating Synthetic Training Set ---\n');
% Target proportions for training: 35% Normal, 40% Subdiffusive, 25% Superdiffusive
propND  = 0.35;
propSub = 0.40;
propSup = 0.25;

% Generate fBM trajectories using the Davies-Harte algorithm
% This call assumes 'step_synthetic' uses the proportions and params defined above
step_synthetic; 

% --- Step 2.1: Integrate Vesicle Diameter into Training Data ---
% To test if diameter influences classification, we bootstrap real 
% diameters into the synthetic training set (size-independent control)
validDiameters = T_dim.diameter(~isnan(T_dim.diameter) & T_dim.diameter > 0);
numSyntheticTotal = height(Training_T);

rng(params.rngSeed);
Training_T.Diameter = randsample(validDiameters, numSyntheticTotal, true);
fprintf('Synthetic training set ready with %d samples.\n', numSyntheticTotal);

%% 3. Machine Learning: Random Forest (RF)
fprintf('\n--- Step 3: Training Random Forest Classifier ---\n');

% Feature Matrix: [Alpha, log10(K_alpha), R-squared, Diameter]
featuresTrain = [Training_T.Alpha, log10(Training_T.D), Training_T.Rsqrt, Training_T.Diameter];
labelsTrain   = Training_T.Label;

% Train Random Forest (Bagged Trees)
RF_Model = fitcensemble(featuresTrain, labelsTrain, ...
                      'Method', 'Bag', ...
                      'NumLearningCycles', params.numTrees, ...
                      'Learners', 'Tree');

% Evaluate Out-of-Bag Error
oobError = oobLoss(RF_Model, 'Mode', 'ensemble');
fprintf('RF Out-of-Bag Classification Error: %.2f%%\n', oobError * 100);

% Predict on Experimental Data
% Note: Using T_dim.diameter for consistency with Training_T
featuresReal = [T.Alpha, log10(T.D), T.R2, T_dim.diameter];
T.Predicted_Label_RF = predict(RF_Model, featuresReal);

%% 4. Deep Learning: 1D CNN
fprintf('\n--- Step 4: Training 1D CNN Pipeline ---\n');

% Training logic resides in step_CNN (processes raw MSD curves)
step_CNN; 

% Predict experimental trajectories using CNN
step_classification; 

%% 5. Results Visualization
fprintf('\n--- Step 5: Generating Publication Figures ---\n');

% Figure A: Alpha Distributions
figure('Name', 'Trajectory Characteristics', 'Color', 'w');
subplot(1,2,1);
histogram(T.Alpha, 20, 'FaceColor', [0.2 0.4 0.6]); 
title('Experimental \alpha Distribution');
xlabel('Anomalous Exponent (\alpha)'); ylabel('Frequency');
grid on; axis square;

% Figure B: Predictor Importance (Crucial for the "Size doesn't matter" finding)
subplot(1,2,2);
imp = predictorImportance(RF_Model);
b = bar(imp, 'FaceAlpha', 0.8);
b.FaceColor = 'flat';
b.CData(4,:) = [0.8 0.2 0.2]; % Highlight Diameter in red
set(gca, 'XTickLabel', {'\alpha', 'log_{10}(K_\alpha)', 'R^2', 'Diameter'});
title('Predictor Importance Scores');
ylabel('Importance Score');
grid on; axis square;

% Figure C: Model Consensus (Confusion Matrix)
figure('Name', 'Model Comparison', 'Color', 'w');
cc = confusionchart(categorical(T.Predicted_Label_RF), categorical(T.Predicted_Label_CNN), ...
    'Title', 'Classification Consensus: RF vs. CNN', ...
    'XLabel', 'CNN Prediction', 'YLabel', 'RF Prediction');
cc.DiagonalColor = [0 0.45 0.74];
set(gca, 'FontSize', 11);

%% 6. Data Export
outputFile = fullfile('results', 'classified_trajectories.mat');
save(outputFile, 'T', 'RF_Model', 'params', '-v7.3');
fprintf('\nAnalysis Complete. Results saved to: %s\n', outputFile);