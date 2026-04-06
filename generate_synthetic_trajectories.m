function Tracks = generate_synthetic_trajectories(N_Tracks, numSteps, dt, alpha, D_mean, D_std, xmin, xmax, ymin, ymax)
% GENERATE_SYNTHETIC_TRAJECTORIES Generate 2D trajectories using fBM or Brownian Motion.
%
% This function simulates particle trajectories based on fractional Brownian 
% Motion (fBM) via the Davies-Harte algorithm. It accounts for experimental 
% constraints such as localization noise and varying diffusion coefficients.
%
% Inputs:
%   N_Tracks - Number of trajectories to generate.
%   numSteps - Points per trajectory (scalar or [N_Tracks x 1] vector).
%   dt       - Time interval between points (seconds).
%   alpha    - Anomalous exponent (0 < alpha < 2).
%              alpha = 1.0: Normal diffusion (Brownian).
%              alpha < 1.0: Subdiffusion.
%              alpha > 1.0: Superdiffusion.
%   D_mean   - Mean of the log10(Diffusion Coefficient).
%   D_std    - Std dev of the log10(Diffusion Coefficient).
%   xmin, xmax, ymin, ymax - Bounds for initial particle placement.
%
% Output:
%   Tracks   - Cell array {N_Tracks x 1}, each cell contains [2 x numSteps] matrix.

%% 1. Initialization and Input Validation
if length(numSteps) == 1
    numSteps = repmat(numSteps, N_Tracks, 1);
elseif length(numSteps) ~= N_Tracks
    error('numSteps must be a scalar or match N_Tracks.');
end

% Parameters
localization_error_std = 0.01; % Simulated experimental noise (micrometers)
epsilon = 1e-6;                % Tolerance for alpha = 1.0
lx = xmax - xmin;
ly = ymax - ymin;
H = alpha / 2.0;               % Hurst exponent

% Pre-allocate output
Tracks = cell(N_Tracks, 1);

% Generate Diffusion Coefficients (D) from a Log-Normal distribution
% to match the heterogeneity observed in biological datasets.
rng('shuffle'); % Ensure different results across runs
logD_synthetic = normrnd(D_mean, D_std, N_Tracks, 1);
D_gen = 10.^logD_synthetic;
D_gen(D_gen <= 0) = 1e-9; % Physical constraint: D must be positive

fprintf('Simulating %d tracks (alpha=%.2f, dt=%.3fs)...\n', N_Tracks, alpha, dt);
tic;

%% 2. Trajectory Generation Loop
parfor i = 1:N_Tracks
    n_steps_current = numSteps(i);
    n_increments = n_steps_current - 1;
    D = D_gen(i);

    % Handle edge cases for short trajectories
    if n_steps_current < 1
        Tracks{i} = zeros(2, 0);
        continue;
    elseif n_steps_current == 1
        Tracks{i} = [xmin + rand * lx; ymin + rand * ly];
        continue;
    end

    % 2.1 Initial position (randomly distributed within bounds)
    x0 = xmin + rand * lx;
    y0 = ymin + rand * ly;

    % 2.2 Generate Increments (dx, dy)
    if abs(alpha - 1.0) < epsilon
        % --- Normal Diffusion (Standard Brownian Motion) ---
        % Sigma is derived from the Einstein-Smoluchowski relation: <x^2> = 2D*dt
        sigma_step = sqrt(2 * D * dt);
        
        % Generate independent Gaussian increments for x and y
        dx = sigma_step * randn(1, n_increments);
        dy = sigma_step * randn(1, n_increments);
    else
        % --- Anomalous Diffusion (fractional Brownian Motion) ---
        % The variance of increments for fBM scales with dt^alpha
        sigma2_1D = 2 * D * (dt^alpha);
        
        % Generate fGN sequences using Davies-Harte algorithm
        dx = generate_fgn_daviesharte(n_increments, H, sigma2_1D);
        dy = generate_fgn_daviesharte(n_increments, H, sigma2_1D);
    end

    % 2.3 Integrate increments to get coordinates
    x = cumsum([x0, dx]);
    y = cumsum([y0, dy]);

    % 2.4 Add Static Localization Noise
    % Simulates the uncertainty in centroid determination during tracking
    x_noisy = x + localization_error_std * randn(size(x));
    y_noisy = y + localization_error_std * randn(size(y));

    Tracks{i} = [x_noisy; y_noisy];
end

elapsed_time = toc;
fprintf('Simulation complete. Elapsed time: %.2f seconds.\n', elapsed_time);

end