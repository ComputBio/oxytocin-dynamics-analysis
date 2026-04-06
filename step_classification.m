%% CNN Classification Pipeline for Experimental Data
% This script applies the trained 1D-CNN model to the real experimental 
% trajectories stored in table 'T'. 

fprintf('\n--- Starting CNN Prediction on Experimental Data ---\n');

%% 1. Input Verification
if ~exist('T', 'var')
    error('Table "T" not found. Please run data_preparation.m first.');
end

if ~exist('cnnNet', 'var')
    fprintf('CNN model not found in workspace. Loading from models/...\n');
    load(fullfile('models', 'trained_CNN_model.mat'));
end

%% 2. Preprocessing Experimental MSD Curves
% It is vital that these curves are normalized and sized identically to 
% the synthetic curves used during training.

msd_curves_real = T.TAMSD;
numReal = height(T);
XReal_processed = cell(numReal, 1);

% Fixed length derived from experimental configuration (Frames - 1)
fixedLength = params.numSteps - 1; 

% Identify empty or invalid tracks
empty_idx = cellfun(@(x) isempty(x) || all(isnan(x)), msd_curves_real);

fprintf('Processing %d experimental trajectories...\n', numReal);

for i = 1:numReal
    if empty_idx(i)
        XReal_processed{i} = zeros(1, fixedLength); % Placeholder for invalid tracks
        continue;
    end
    
    seq = msd_curves_real{i}(:)'; % Ensure row vector

    % --- Step A: Standardize Length ---
    if length(seq) > fixedLength
        seq = seq(1:fixedLength);
    elseif length(seq) < fixedLength
        % Pad with the last valid value (maintains the diffusion trend)
        seq = [seq, repmat(seq(end), 1, fixedLength - length(seq))];
    end

    % --- Step B: Min-Max Normalization ---
    % Identical to training: focuses the CNN on the 'shape' of the MSD curve.
    minVal = min(seq);
    maxVal = max(seq);
    rangeVal = maxVal - minVal;

    if rangeVal < 1e-12
        XReal_processed{i} = zeros(1, fixedLength);
    else
        XReal_processed{i} = (seq - minVal) / rangeVal;
    end
end

%% 3. Label Prediction
fprintf('Predicting labels via 1D-CNN...\n');

% Set batch size (default to 128 if options not available)
bSize = 128;
if exist('options', 'var'), bSize = options.MiniBatchSize; end

% Run Classification
YPred_CNN = classify(cnnNet, XReal_processed, 'MiniBatchSize', bSize);

% Re-assign invalid tracks to a 'Missing' or 'NaN' category if necessary
if any(empty_idx)
    YPred_CNN(empty_idx) = categorical({'Invalid'});
    fprintf('Warning: %d tracks were invalid and assigned "Invalid" label.\n', sum(empty_idx));
end

% Store results back into the main table
T.Predicted_Label_CNN = YPred_CNN;

%% 4. Results Visualization
figure('Name', 'CNN Prediction Summary', 'Color', 'w');
h = histogram(T.Predicted_Label_CNN, 'DisplayOrder', 'descend');
h.FaceColor = [0.2 0.6 0.4];
title('CNN Predicted Motion Modes (Experimental Data)');
xlabel('Diffusion Classification');
ylabel('Number of Trajectories');
grid on;

% Print summary to console
summary(T.Predicted_Label_CNN);
fprintf('CNN classification results successfully added to table T.\n');