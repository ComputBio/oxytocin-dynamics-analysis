%% Synthetic Trajectory Generation Pipeline
% This script generates 1D/2D synthetic trajectories for three diffusion modes:
% Normal Diffusion (ND), Subdiffusion (sub), and Superdiffusion (SUP).
% It uses the Davies-Harte algorithm via 'generate_synthetic_trajectories'.

fprintf('\n--- Starting Synthetic Data Generation ---\n');

% 1. Define Mode Configurations
% Each mode has a target Alpha, a proportion of the total set, and a spatial bound
modes = {'ND', 'sub', 'SUP'};
targetAlphas = [1.0, 0.3, 1.7];
proportions  = [N_nd, N_sub, N_sup];
plotIndices  = [2, 3, 4]; % Subplot positions (Matching the Main script's 2x2 grid)

% Spatial bounds (Adjusted based on your specific experimental constraints)
% Format: [x_min, x_max, y_min, y_max]
bounds = { [0, 20, 0, 20], [0, 1, 0, 1], [0, 1, 0, 1] };

% Cell array to store tables for each mode
modeTables = cell(1, length(modes));

for m = 1:length(modes)
    currentMode = modes{m};
    alphaVal    = targetAlphas(m);
    nTracks     = floor(N_synthetic * proportions(m));
    b           = bounds{m};
    
    fprintf('Generating %d tracks for Mode: %s (Target alpha: %.1f)...\n', ...
            nTracks, currentMode, alphaVal);

    % Pre-allocate results for performance
    T_Length = zeros(nTracks, 1);
    T_TAMSD  = cell(nTracks, 1);
    T_Alpha  = zeros(nTracks, 1);
    T_D      = zeros(nTracks, 1);
    T_Rsq    = zeros(nTracks, 1);

    % Generate trajectories using Davies-Harte fBM algorithm
    synthTracks = generate_synthetic_trajectories(nTracks, numSteps, timestep, ...
                  alphaVal, logD_mean_real, logD_std_real, b(1), b(2), b(3), b(4));

    % Process trajectories in parallel
    parfor i = 1:nTracks
        track = synthTracks{i};
        T_Length(i) = size(track, 2);

        % MSD analysis: [time, msd, alpha, D, R-squared]
        % Assumes track is [2 x N] matrix
        [~, msd, alpha_est, D_est, Rsq_est] = MSD(track(1,:), track(2,:), timestep);
       
        T_TAMSD{i} = msd;
        T_D(i)     = D_est;
        T_Alpha(i) = alpha_est;
        T_Rsq(i)   = Rsq_est;
    end

    % Create table for this specific mode
    labels = repmat({currentMode}, nTracks, 1);
    modeTables{m} = table(T_Length, T_TAMSD, T_Alpha, T_D, T_Rsq, labels, ...
        'VariableNames', {'Length', 'MSD_Curve', 'Alpha', 'D', 'Rsqrt', 'Label'});

    % Visualization
    subplot(2, 2, plotIndices(m));
    histogram(T_Alpha, 50, 'FaceColor', [0.4 0.4 0.4], 'EdgeAlpha', 0.5); 
    title(['Synthetic ', currentMode, ' \alpha Distribution']);
    xlabel('Estimated \alpha'); ylabel('Frequency');
    grid on;
end

%% 2. Merge and Export
% Concatenate all mode tables into one master Training Table
Training_T = vertcat(modeTables{:});

% Reporting Summary
fprintf('\n--- Synthetic Dataset Summary ---\n');
groupCounts = groupsummary(Training_T, 'Label');
disp(groupCounts(:, {'Label', 'GroupCount'}));

% Save the generated training data
outputFolder = 'results';
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
savePath = fullfile(outputFolder, 'Real_Training_Data.mat');

save(savePath, 'Training_T', '-v7.3');
fprintf('Synthetic training dataset saved to: %s\n', savePath);