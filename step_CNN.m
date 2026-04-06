%% 1D-CNN Training Pipeline for Trajectory Classification
% This script trains a 1D Convolutional Neural Network to classify 
% trajectories based on the temporal patterns of their MSD curves.

fprintf('\n--- Starting CNN Training Pipeline ---\n');

%% 1. Data Preprocessing
msd_curves_cell = Training_T.MSD_Curve;
labels = Training_T.Label;

% Remove empty or invalid entries
valid_idx = cellfun(@(x) ~isempty(x) && ~any(isnan(x)), msd_curves_cell);
if any(~valid_idx)
    fprintf('Removing %d invalid MSD curves...\n', sum(~valid_idx));
    msd_curves_cell = msd_curves_cell(valid_idx);
    labels = labels(valid_idx);
end

% Set fixed sequence length based on experimental parameters (from main.m)
fixedLength = params.numSteps - 1; 
numObservations = numel(msd_curves_cell);
sequences_processed = cell(numObservations, 1);

fprintf('Standardizing and Normalizing %d sequences (Length: %d)...\n', ...
        numObservations, fixedLength);

for i = 1:numObservations
    seq = msd_curves_cell{i}(:)'; % Ensure row vector

    % Handle sequence length (Truncate or Pad with nearest value)
    if length(seq) > fixedLength
        seq = seq(1:fixedLength);
    elseif length(seq) < fixedLength
        % Pad with the last valid value to maintain the "plateau" or "trend"
        seq = [seq, repmat(seq(end), 1, fixedLength - length(seq))];
    end

    % Min-Max Normalization: Scales curve between [0, 1]
    % This focuses the CNN on the 'shape' (curvature) rather than absolute magnitude.
    minVal = min(seq);
    maxVal = max(seq);
    rangeVal = maxVal - minVal;
    
    if rangeVal < 1e-12
        sequences_processed{i} = zeros(1, fixedLength);
    else
        sequences_processed{i} = (seq - minVal) / rangeVal;
    end
end

% --- Prepare Labels and Split Data ---
Y_cnn = categorical(labels);
classNames = categories(Y_cnn);
numClasses = numel(classNames);

rng(params.rngSeed); % For reproducibility
cvp = cvpartition(Y_cnn, 'HoldOut', 0.20); % 80/20 Split

XTrain = sequences_processed(training(cvp));
YTrain = Y_cnn(training(cvp));
XVal   = sequences_processed(test(cvp));
YVal   = Y_cnn(test(cvp));

%% 2. CNN Architecture Definition
% The architecture uses 1D convolutions to extract temporal features, 
% followed by Global Average Pooling to ensure size-invariance.

layers = [
    sequenceInputLayer(1, 'Name', 'input', 'Normalization', 'none')

    convolution1dLayer(5, 16, 'Padding', 'causal', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling1dLayer(3, 'Stride', 2, 'Padding', 'same', 'Name', 'pool1')

    convolution1dLayer(5, 32, 'Padding', 'causal', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling1dLayer(3, 'Stride', 2, 'Padding', 'same', 'Name', 'pool2')

    convolution1dLayer(3, 64, 'Padding', 'causal', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    % Global Average Pooling reduces the sequence to a feature vector 
    % while maintaining robustness against local noise.
    globalAveragePooling1dLayer('Name', 'gapool')

    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% 3. Training Configuration
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'auto'); % Use GPU if available

%% 4. Model Training
fprintf('Training 1D-CNN on synthetic data...\n');
[cnnNet, trainInfo] = trainNetwork(XTrain, YTrain, layers, options);

%% 5. Evaluation & Export
YPred = classify(cnnNet, XVal);
accuracy = mean(YPred == YVal);
fprintf('CNN Validation Accuracy: %.2f%%\n', accuracy * 100);

% Visualization: Confusion Matrix
figure('Name', 'CNN Performance', 'Color', 'w');
confusionchart(YVal, YPred, 'Title', 'CNN Validation Results (Synthetic Data)');
set(gca, 'FontSize', 12);

% Save the trained model
if ~exist('models', 'dir'), mkdir('models'); end
save(fullfile('models', 'trained_CNN_model.mat'), 'cnnNet', 'params');
fprintf('CNN model saved to models/trained_CNN_model.mat\n');