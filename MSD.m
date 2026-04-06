function [tVec, msd, alpha, K_alpha, Rsquared] = MSD(x, y, dt)
% MSD Calculates the Time-Averaged Mean Squared Displacement (TAMSD).
%
% This function computes the TAMSD for a 2D trajectory and estimates 
% the anomalous exponent (alpha) and generalized diffusion coefficient (K_alpha)
% by fitting the power-law relation: MSD(tau) = 4 * K_alpha * tau^alpha.
%
% Inputs:
%   x, y    - Vectors of particle coordinates.
%   dt      - Time interval between frames (seconds).
%
% Outputs:
%   tVec     - Vector of time lags (tau).
%   msd      - Calculated TAMSD values for each lag.
%   alpha    - Anomalous exponent (slope of log-log fit).
%   K_alpha  - Generalized diffusion coefficient (intercept of log-log fit).
%   Rsquared - Goodness of fit for the power-law model.

%% 1. Calculate Time-Averaged MSD
N = length(x);
maxLag = N - 1;
msd = zeros(1, maxLag);
tVec = (1:maxLag) * dt;

for n = 1:maxLag
    % Squared displacements for lag n
    dx2 = (x(1+n:end) - x(1:end-n)).^2;
    dy2 = (y(1+n:end) - y(1:end-n)).^2;
    
    % Average over all possible pairs for this lag
    msd(n) = mean(dx2 + dy2);
end

%% 2. Anomalous Exponent Estimation (Log-Log Fit)
% In SPT analysis, it is standard practice to fit only the first ~25% of 
% the MSD curve, as the statistics at higher lags are based on fewer 
% data points and are highly susceptible to noise.
fitRange = 1:floor(maxLag * 0.25);

% Filter for positive MSD values to avoid log(0) errors
validIdx = fitRange(msd(fitRange) > 0);

if length(validIdx) < 2
    % Not enough data points to perform a fit
    alpha = NaN; K_alpha = NaN; Rsquared = 0;
    return;
end

% Transform to log-log space
logTau = log(tVec(validIdx));
logMSD = log(msd(validIdx));

% Linear regression: log(MSD) = alpha * log(tau) + log(4 * K_alpha)
p = polyfit(logTau, logMSD, 1);

alpha = p(1); 
% For 2D diffusion: Intercept = log(4 * K_alpha)
K_alpha = exp(p(2)) / 4;

%% 3. Calculate Goodness of Fit (R^2)
yFit = polyval(p, logTau);
yMean = mean(logMSD);

SS_res = sum((logMSD - yFit).^2);
SS_tot = sum((logMSD - yMean).^2);

if SS_tot > 1e-12
    Rsquared = 1 - (SS_res / SS_tot);
else
    Rsquared = 0; % Data is essentially a single point or constant
end

% Constraint: R2 cannot be negative (can happen with extremely poor fits)
Rsquared = max(0, Rsquared);

end