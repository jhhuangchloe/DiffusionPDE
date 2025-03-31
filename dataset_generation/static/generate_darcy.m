function generate(N, S)
    % Set default values if not provided
    for round = 1:7
        if nargin < 1
            N = 10000; % Default number of generations
        end
        if nargin < 2
            S = 128; % Default resolution
        end
    
        % Preallocate arrays to store the generated data
        lognorm_a_data = zeros(N, S, S);
        thresh_a_data = zeros(N, S, S);
        lognorm_p_data = zeros(N, S, S);
        thresh_p_data = zeros(N, S, S);
    
        % Parameters for Gaussian Random Field (GRF)
        alpha = 2;
        tau = 3;
    
        % Forcing function, f(x) = 1
        f = ones(S, S);
    
        for i = 1:N
            % Generate random coefficients from N(0,C)
            norm_a = GRF(alpha, tau, S);
    
            % Exponentiate it to ensure a(x) > 0 (Lognormal)
            lognorm_a = exp(norm_a);
    
            % Thresholding to achieve ellipticity
            thresh_a = zeros(S, S);
            thresh_a(norm_a >= 0) = 12;
            thresh_a(norm_a < 0) = 3;
    
            % Solve PDE: -div(a(x)*grad(p(x))) = f(x)
            lognorm_p = solve_gwf(lognorm_a, f);
            thresh_p = solve_gwf(thresh_a, f);
    
            % Store the generated data
            lognorm_a_data(i, :, :) = lognorm_a;
            thresh_a_data(i, :, :) = thresh_a;
            lognorm_p_data(i, :, :) = lognorm_p;
            thresh_p_data(i, :, :) = thresh_p;
        end
    
        % Ensure the data folder exists
        if ~exist('data', 'dir')
           mkdir('data')
        end
    
        % Save the data in a .mat file
        filename = sprintf('data/%d-%d-%d_%d.mat', N, S, S, round);
        save(filename, 'lognorm_a_data', 'thresh_a_data', 'lognorm_p_data', 'thresh_p_data', '-v7.3');
    end
end
