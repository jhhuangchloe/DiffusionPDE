function generateHelmholtzData(N, S, k)
    for round = 1:2
        if nargin < 1
            N = 1; % Default number of generations
        end
        if nargin < 2
            S = 128; % Default resolution
        end
        if nargin < 3
            k = 1; % Default k
        end
        f_data = zeros(N, S, S);
        psi_data = zeros(N, S, S);
        
        h = 1 / (S - 1);  
        x = linspace(0, 1, S);
        y = linspace(0, 1, S);
        [X, Y] = meshgrid(x, y);
        
        e = ones(S, 1);
        L = spdiags([e -2*e e], -1:1, S, S) / h^2;

        L(1, :) = 0; L(1, 1) = 1; 
        L(S, :) = 0; L(S, S) = 1; 
        L_full = kron(speye(S), L) + kron(L, speye(S));
        
        % Parameters for GRF
        alpha = 2;
        tau = 3;
        
        for i = 1:N
            f = GRF(alpha, tau, S);
            f(1, :) = 0; f(S, :) = 0; f(:, 1) = 0; f(:, S) = 0;
            f_data(i, :, :) = f;
            
            A = L_full + k^2 * speye(S^2);
            f_vector = reshape(f, [S^2, 1]);
            psi_vector = A \ f_vector;
            psi = reshape(psi_vector, [S, S]);
            
            psi_data(i, :, :) = psi;
            figure;
            subplot(1, 2, 1);
            imagesc(psi);
            title('\psi');
            colorbar;
            axis square;
            
            subplot(1, 2, 2);
            imagesc(f);
            title('f');
            colorbar;
            axis square;
            
            psi_vector = reshape(psi, [S^2, 1]);
            computed_f_vector = L_full * psi_vector + k^2 * psi_vector;
            computed_f = reshape(computed_f_vector, [S, S]);

            figure;
            subplot(1, 2, 1);
            imagesc(computed_f);
            title('Computed f');
            colorbar;
            axis square;
            
            subplot(1, 2, 2);
            imagesc(f);
            title('Original f');
            colorbar;
            axis square;
            
            difference = computed_f - f;
            figure;
            imagesc(difference);
            title('Difference (Computed f - Original f)');
            colorbar;
            axis square;

            top_edge = psi(1, :);
            bottom_edge = psi(end, :);
            left_edge = psi(:, 1);
            right_edge = psi(:, end);
            
            disp('Top edge values:');
            disp(top_edge);
            
            disp('Bottom edge values:');
            disp(bottom_edge);
            
            disp('Left edge values:');
            disp(left_edge);
            
            disp('Right edge values:');
            disp(right_edge);

            tolerance = 1e-6; 
            is_zero_top = all(abs(top_edge) < tolerance);
            is_zero_bottom = all(abs(bottom_edge) < tolerance);
            is_zero_left = all(abs(left_edge) < tolerance);
            is_zero_right = all(abs(right_edge) < tolerance);
            disp(['Top edge is zero: ', num2str(is_zero_top)]);
            disp(['Bottom edge is zero: ', num2str(is_zero_bottom)]);
            disp(['Left edge is zero: ', num2str(is_zero_left)]);
            disp(['Right edge is zero: ', num2str(is_zero_right)]);
            
            if is_zero_top && is_zero_bottom && is_zero_left && is_zero_right
                disp('All edges are zero. Boundary condition verified.');
            else
                disp('At least one edge is not zero. Boundary condition not satisfied.');
            end


        end
        % Save the dataset
        if ~exist('data', 'dir')
            mkdir('data');
        end
        % filename = sprintf('data/helmholtz_%d-%d-%d_%d.mat', N, S, S, round);
        % save(filename, 'f_data', 'psi_data');
    end
end