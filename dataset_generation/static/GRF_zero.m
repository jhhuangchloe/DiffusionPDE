function U = GRF_zero(alpha, tau, s)
    % Random variables in KL expansion
    xi = normrnd(0, 1, s, s);
    
    % Define the (square root of) eigenvalues of the covariance operator
    [K1, K2] = meshgrid(0:s-1, 0:s-1);
    coef = tau^(alpha-1) * (pi^2 * (K1.^2 + K2.^2) + tau^2).^(-alpha/2);
    
    % Construct the KL coefficients
    L = s * coef .* xi;
    L(1,1) = 0;  % Remove the mean component to maintain zero mean
    
    % Perform inverse discrete cosine transform
    U = idct2(L);
    
    % Apply a window function to smoothly bring the edges to zero
    [window_x, window_y] = meshgrid(linspace(-1, 1, s), linspace(-1, 1, s));
    window = (0.75 * (1 + cos(pi * window_x))) .* (0.75 * (1 + cos(pi * window_y)));
    U = U .* window;
end
