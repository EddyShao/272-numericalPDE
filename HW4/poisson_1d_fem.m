
function A = poisson_fem_mat_1d(N)
    e = ones(N-1, 1);  % Block of ones for the neighboring diagonal
    A = spdiags([-e 2*e -e], -1:1, N-1, N-1);  % 1D Poisson matrix for each block
end



f = @(x) - (4*x^4 - 4*x^3 + 10*x^2 - 6*x + 2) * exp(x^2 + 1);

u = @(x) x.*(x-1).* exp(x.^2 + 1);

u_prime = @(x) (2*x^3 - 2*x^2 + 2*x - 1) * exp(x^2 + 1);


function integral = composite_gauss(f, N)
    h = (1-0)/N;
    integral = 0;
    for i=1:N
        midpoint = (i-0.5)*h;
        node_1 = midpoint - h/(2*sqrt(3));
        node_2 = midpoint + h/(2*sqrt(3));
        integral = integral + f(node_1) + f(node_2);
    end
    integral = (h/2)*integral;
end



function phi_i = phi(i, h, x)
    if ((i-1)*h < x) && (x<=i*h)
        phi_i = (x - (i-1)*h)/h;
    elseif (i*h < x) && (x< (i+1)*h)
        phi_i = ((i+1)*h - x)/h;
    else
        phi_i = 0;
    end
end

function phi_i_prime = phi_prime(i, h, x)
    if ((i-1)*h < x) && (x<=i*h)
        phi_i_prime = 1/h;
    elseif (i*h < x) && (x< (i+1)*h)
        phi_i_prime = -1/h;
    else
        phi_i_prime = 0;
    end
end

function u_h = solve_poisson(N, f)
    h = (1 - 0)/ N;
    F = zeros(N-1, 1);
    % x = linspace(0, 1, N+1);
    for i=1:N-1
        F_i = @(x) phi(i, h, x)*f(x);
        F(i, 1) = composite_gauss(F_i, N);
    end
    A = N * poisson_fem_mat_1d(N);
    u_h = A \ F;
end

function u_true = solve_u_true(N, u)
    x = linspace(0, 1, N+1);
    u_true = u(x(2:N))';
end

N_int = 1000; % points for integration



% 
% Ns = [10, 20, 40, 80, 160];
% fprintf('N\t\t||u_h - u||_2\n');
% fprintf('----------------------------------\n');
% for idx = 1:length(Ns)
%     N = Ns(idx);
%     h = 1/N;
%     u_h = solve_poisson(N, f);
% 
%     u_h_func = @(x) sum(arrayfun(@(i) u_h(i, 1) * phi(i, h, x), 1:N-1));
%     u_h_func_prime = @(x) sum(arrayfun(@(i) u_h(i, 1) * phi_prime(i, h, x), 1:N-1));
%     integrand_1 = @(x) (u_h_func(x) - u(x))^2;
%     integrand_2 = @(x) (u_h_func_prime(x) - u_prime(x))^2;
% 
%     fprintf('%3d\t\t%.9f\n', Ns(idx), ...
%         composite_gauss(integrand_1, N_int)^.5);
% end
% 
% fprintf('N\t\t||u_h_prime - u_prime||_2\n');
% fprintf('----------------------------------\n');
% for idx = 1:length(Ns)
%     N = Ns(idx);
%     h = 1/N;
%     u_h = solve_poisson(N, f);
% 
%     u_h_func = @(x) sum(arrayfun(@(i) u_h(i, 1) * phi(i, h, x), 1:N-1));
%     u_h_func_prime = @(x) sum(arrayfun(@(i) u_h(i, 1) * phi_prime(i, h, x), 1:N-1));
%     integrand_1 = @(x) (u_h_func(x) - u(x))^2;
%     integrand_2 = @(x) (u_h_func_prime(x) - u_prime(x))^2;
% 
%     fprintf('%3d\t\t%.9f\n', Ns(idx), ...
%         composite_gauss(integrand_2, N_int)^.5);
% end
% 
% 
% fprintf('N\t\t||u_h - u||_{H^1}\n');
% fprintf('----------------------------------\n');
% for idx = 1:length(Ns)
%     N = Ns(idx);
%     h = 1/N;
%     u_h = solve_poisson(N, f);
% 
%     u_h_func = @(x) sum(arrayfun(@(i) u_h(i, 1) * phi(i, h, x), 1:N-1));
%     u_h_func_prime = @(x) sum(arrayfun(@(i) u_h(i, 1) * phi_prime(i, h, x), 1:N-1));
%     integrand_1 = @(x) (u_h_func(x) - u(x))^2;
%     integrand_2 = @(x) (u_h_func_prime(x) - u_prime(x))^2;
% 
%     norm_squared_sum = composite_gauss(integrand_1, N_int) + composite_gauss(integrand_2, N_int);
% 
%     fprintf('%3d\t\t%.9f\n', Ns(idx), ...
%         norm_squared_sum^.5);
% end

Ns = 10:20:200;  % Calculate the inverse and convert to integers
hs = 1./ Ns;
errors_1 = zeros(length(Ns));
errors_2 = zeros(length(Ns));
errors_3 = zeros(length(Ns));
for idx = 1:length(Ns)
    N = Ns(idx);
    h = 1/N;
    u_h = solve_poisson(N, f);

    u_h_func = @(x) sum(arrayfun(@(i) u_h(i, 1) * phi(i, h, x), 1:N-1));
    u_h_func_prime = @(x) sum(arrayfun(@(i) u_h(i, 1) * phi_prime(i, h, x), 1:N-1));
    integrand_1 = @(x) (u_h_func(x) - u(x))^2;
    integrand_2 = @(x) (u_h_func_prime(x) - u_prime(x))^2;
    errors_1(idx) = composite_gauss(integrand_1, N_int);
    errors_2(idx) = composite_gauss(integrand_2, N_int);
    errors_3(idx) = (errors_1(idx) + errors_2(idx))^.5;
    errors_1(idx) = errors_1(idx)^.5;
    errors_2(idx) = errors_2(idx)^.5;
end

% Take logarithm of h and errors, avoiding log(0) issues
log_h = log(hs);


fprintf('L2');
log_errors = log(errors_1);

% Remove NaN and -Inf values (if any)
valid_indices = ~isnan(log_errors) & ~isinf(log_errors) & ~isnan(log_h) & ~isinf(log_h);
log_h_valid = log_h(valid_indices);
log_errors_valid = log_errors(valid_indices);

% Perform linear regression
p = polyfit(log_h_valid, log_errors_valid, 1); % p(1) is the slope m, p(2) is the intercept b

% Display results
m = p(1); % slope
b = p(2); % intercept
fprintf('Slope (m): %.4f\n', m);
fprintf('Intercept (b): %.4f\n', b);

fprintf('energy');

log_errors = log(errors_2);

% Remove NaN and -Inf values (if any)
valid_indices = ~isnan(log_errors) & ~isinf(log_errors) & ~isnan(log_h) & ~isinf(log_h);
log_h_valid = log_h(valid_indices);
log_errors_valid = log_errors(valid_indices);

% Perform linear regression
p = polyfit(log_h_valid, log_errors_valid, 1); % p(1) is the slope m, p(2) is the intercept b

% Display results
m = p(1); % slope
b = p(2); % intercept
fprintf('Slope (m): %.4f\n', m);
fprintf('Intercept (b): %.4f\n', b);


% figure;  % Create a new figure window
% clf;  % Clears the current figure
% 
% % Plot the first and second sets of errors with specific colors
% loglog(hs, errors_1, '-o', hs, errors_2, '-o', 'LineWidth', 2);  % Blue line with circle markers for L2 error
% legend('L2', 'Energy', 'Interpreter', 'latex', 'Location', 'best');  % Correct legend
% 
% % Add legend for both plotted lines with correct labels
% % legend({'$\|u-u_{h}\|_2$', '$\|u^{''}- u_{h}^{''}\|_{2}$'}, 'Interpreter', 'latex', 'Location', 'best');  % Correct legend
% % Turn off the toolbar in the current axes
% ax = gca;
% ax.Toolbar.Visible = 'off';
% 
% % Set the x-axis and y-axis labels with LaTeX interpreter
% xlabel('$h$', 'Interpreter', 'latex', 'Color', 'k');  % Use LaTeX interpreter for 'h' and set color to black
% ylabel('$\|u - u_h\|_{2}$', 'Interpreter', 'latex', 'Color', 'k');  % Use LaTeX interpreter for the norm expression and set color to black
% 
% % Add grid and title
% grid on;
% title('Error Convergence Plot', 'Interpreter', 'latex');  % Set title with LaTeX interpreter
% 
% hold off;
% saveas(gcf, 'error_loglog_L2_energy.png')
