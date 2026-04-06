%% Data Preparation Pipeline: Trajectory Extraction and MSD Analysis
% 
% This script flattens the experimental data structure into a unified table
% and computes the Time-Averaged Mean Squared Displacement (TAMSD) and 
% anomalous diffusion parameters (Alpha, D, R^2) for each trajectory.

clearvars; close all; clc;

%% 1. Load Data Structure
% The script expects a 'data_struct.mat' containing a struct 'allData' 
% where each field represents an experimental group/sheet.
structPath = fullfile('data', 'data_struct.mat');

if exist(structPath, 'file')
    fprintf('Loading existing data structure: %s\n', structPath);
    load(structPath);
else
    fprintf('Data structure not found. Attempting to import from Excel...\n');
    excel_load; % Ensure excel_load.m is in your path
end

%% 2. Flatten Structure and Extract Trajectories
sheetNames = fieldnames(allData);
totalPossibleTracks = 0;

% First pass: Count total trajectories for pre-allocation
for i = 1:length(sheetNames)
    dataTable = allData.(sheetNames{i});
    % Assumes Column 1 is Time, and remaining columns are X-Y pairs
    totalPossibleTracks = totalPossibleTracks + (size(dataTable, 2) - 1) / 2;
end

% Pre-allocate cell arrays for performance
X = cell(totalPossibleTracks, 1);
Y = cell(totalPossibleTracks, 1);
cellname = cell(totalPossibleTracks, 1);
group = strings(totalPossibleTracks, 1);

trackIdx = 0;
fprintf('--- Extracting Trajectories ---\n');

for i = 1:length(sheetNames)
    currentSheetName = sheetNames{i};
    dataTable = allData.(currentSheetName);
    varNames = dataTable.Properties.VariableNames;
    
    numTracksInSheet = (size(dataTable, 2) - 1) / 2;
    fprintf('Processing Group: %-15s | Tracks: %d\n', currentSheetName, numTracksInSheet);

    for k = 1:numTracksInSheet
        trackIdx = trackIdx + 1;
        
        % Extract X and Y coordinates (skipping the time column at index 1)
        X{trackIdx} = table2array(dataTable(:, 2*k));
        Y{trackIdx} = table2array(dataTable(:, 2*k+1));
        
        % Clean up variable names (removing _x or _y suffixes)
        varName = varNames{2*k};
        baseName = regexprep(varName, '_[xyXY]$', ''); 
        
        cellname{trackIdx} = baseName;
        group(trackIdx) = string(currentSheetName);
    end
end

% Trim pre-allocated arrays if necessary
X = X(1:trackIdx);
Y = Y(1:trackIdx);
cellname = cellname(1:trackIdx);
group = group(1:trackIdx);

fprintf('Total trajectories extracted: %d\n', trackIdx);

%% 3. Calculate MSD and Diffusion Parameters
% Uses 'parfor' to speed up the analysis of individual trajectories.
fprintf('\n--- Calculating TAMSD and Anomalous Exponents ---\n');

TAMSD = cell(trackIdx, 1);
Alpha = zeros(trackIdx, 1);
D     = zeros(trackIdx, 1);
R2    = zeros(trackIdx, 1);

% Ensure Parallel Pool is running for parfor
if isempty(gcp('nocreate'))
    parpool('local'); 
end

parfor i = 1:trackIdx
    % MSD function calculates: [time, msd, alpha, DiffusionCoef, R_squared]
    % Replace '1' with your actual frame interval if different.
    [~, msd, alpha, Dif, Rsquared] = MSD(X{i}, Y{i}, 1);
    
    TAMSD{i} = msd;
    D(i)     = Dif;
    R2(i)    = Rsquared;
    Alpha(i) = alpha;
end

%% 4. Construct Table and Save
T = table(group, cellname, X, Y, TAMSD, Alpha, D, R2, ...
    'VariableNames', {'Group', 'CellName', 'X', 'Y', 'TAMSD', 'Alpha', 'D', 'R2'});

% Save result in the data folder
outputFile = fullfile('data', 'prepared_data.mat');
save(outputFile, 'T', '-v7.3');

% Summary Display
fprintf('\n--- Preparation Complete ---\n');
fprintf('Mean Alpha: %.4f\n', mean(Alpha, 'omitnan'));
fprintf('Mean D:     %.4f\n', mean(D, 'omitnan'));
fprintf('Results saved to: %s\n', outputFile);