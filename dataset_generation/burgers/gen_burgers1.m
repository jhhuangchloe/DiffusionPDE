for round = 1:1
    % number of realizations to generate
    N = 3;
    
    % parameters for the Gaussian random field
    gamma = 2.5;
    tau = 7;
    sigma = 7^(2);
    
    % viscosity
    visc = 1/100;
    
    % grid size
    s = 128;
    steps = 127;
    
    
    input = zeros(N, s);
    if steps == 1
        output = zeros(N, s);
    else
        output = zeros(N, steps, s);
    end
    
    tspan = linspace(0,1,steps+1);
    x = linspace(0,1,s+1);
    for j=1:N
        u0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
        u = burgers1(u0, tspan, s, visc);
        
        u0eval = u0(x);
        input(j,:) = u0eval(1:end-1);
        
        if steps == 1
            output(j,:) = u.values;
        else
            for k=2:(steps+1)
                output(j,k,:) = u{k}.values;
            end
        end
        
        output(j,1,:)=input(j,:);
        % disp(j);
        figure;
        imagesc(tspan(1:end), x(1:end), squeeze(output(j,1:end,:)).');
        colorbar;
        xlabel('Time');
        ylabel('Space');
        title('Burgers Equation Evolution');
    
    end
    
    % output = output(:, 2:end, :);
    % tspan = tspan(2:end);
    % x = x(2:end);
    
    % filename = sprintf('data/burger_%d-%d-%d_%d.mat', N, s, steps+1, round);
    % save(filename, 'output', 'input');

end
