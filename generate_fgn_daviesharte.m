function fgn_sequence = generate_fgn_daviesharte(N, H, sigma2)
% Generates fractional Gaussian noise (fGn) using the Davies-Harte method.
% INPUTS:
%   N      - Desired length of the fGn sequence (number of increments).
%   H      - Hurst exponent (0 < H < 1). alpha = 2*H.
%   sigma2 - Desired variance of the increments, sigma2 = <(X(t+dt)-X(t))^2>.
%
% OUTPUT:
%   fgn_sequence - A sequence of N correlated Gaussian random numbers (fGn).

    if H <= 0 || H >= 1
        % Allow H=1 (alpha=2), though results might be numerically sensitive.
        if abs(H-1.0) < 1e-9
             warning('generate_fgn_daviesharte:HLimit', 'H=1 requested. Near ballistic limit.');
        else
            error('Hurst exponent H must be between 0 and 1.');
        end
    end
    if N <= 0
        fgn_sequence = [];
        return;
    end

    if N == 1
        M = 2; % Minimal M for one increment
    else
        M = 2^nextpow2(2*N); % M is power of 2, M >= 2*N
    end

    % Calculate ACVF gamma(k) for lags k = 0, 1, ..., M/2
    % gamma(k) = 0.5 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H)) for standard fGn
    k_lags = (0:(M/2))'; % Use column vector for clarity if needed
    gamma = 0.5 * (abs(k_lags + 1).^(2 * H) ...
                  - 2 * abs(k_lags).^(2 * H) ...
                  + abs(k_lags - 1).^(2 * H));
    
    % first row R of the Circulant Covariance Matrix 
    R = zeros(1, M);
    R(1) = gamma(1); % k=0
    if M >= 2
        R(2:(M/2 + 1)) = gamma(2:(M/2 + 1)); % k=1 to M/2
        R((M/2 + 2):M) = gamma(M/2:-1:2);    % k=M/2-1 down to 1
    end

    lambda = real(fft(R)); % Eigenvalues of the circulant matrix
    min_lambda = min(lambda);
    lambda(lambda < 0) = 0;
    
    % Gaussian Noise in Frequency Domain
    Z1 = randn(1, M);
    Z2 = randn(1, M);
    W = sqrt(lambda/ (2*M)) .* (Z1 + 1i*Z2); % Generate in freq domain
                                             % Division by M from ifft definition? Or 2*M? 

    %Transform to Time Domain 
    f = fft(W); 

    % The first N elements of the real part form the fGn sequence.
    fgn_std = real(f(1:N));

    %  Scale to desired variance sigma2 
    fgn_sequence = fgn_std * sqrt(sigma2);
end