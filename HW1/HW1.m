% Note that since the boundary is 0 here, we don't have any correction term

function A = poisson_matrix(N)
    e = ones(N-1, 1);  % Block of ones for the neighboring diagonals
    I = speye(N-1);    % Identity matrix
    T = spdiags([e -2*e e], -1:1, N-1, N-1);  % 1D Poisson matrix for each block
    A = kron(I, T) + kron(T, I); % 2D Poisson matrix
    A = -A;
end

f = @(x) 2*pi^2* sin(pi*x(1))*sin(pi*x(2)) + 5*pi^2* sin(pi*x(1))*sin(2*pi*x(2));

u = @(x) sin(pi * x(1)) * sin(pi * x(2)) + sin(pi * x(1)) * sin(2 * pi * x(2));

function xy = cartesianProduct(x, y)
    [X, Y] = ndgrid(x, y);
    xy = [X(:), Y(:)];
end 



function u_h = solve_poisson(N, f)
    h = 1 / N;
    x = linspace(0, 1, N+1);
    y = linspace(0, 1, N+1);
    xy = cartesianProduct(x(2:N), y(2:N));
    f_value = arrayfun(@(i) f(xy(i, :)), 1:size(xy, 1))';
    A = (1/h^2) * poisson_matrix(N);
    u_h = A \ f_value;
end

function u_true = solve_u_true(N, u)
    x = linspace(0, 1, N+1);
    y = linspace(0, 1, N+1);
    xy = cartesianProduct(x(2:N), y(2:N));
    u_true = arrayfun(@(i) u(xy(i, :)), 1:size(xy, 1))';
end


% Display results in a table
Ns = [10, 20, 40, 80];
fprintf('N\t\t||u_h - u||_∞\n');
fprintf('----------------------------------\n');
for idx = 1:length(Ns)
    fprintf('%.4f\t\t%.6f\n', Ns(idx), ...
        max(abs(solve_poisson(Ns(idx), f) - solve_u_true(Ns(idx), u))));
end


Ns = 10:1:300;  % Calculate the inverse and convert to integers
hs = 1 ./ Ns;
errors = zeros(length(Ns));
for idx = 1:length(Ns)
    errors(idx) = max(abs(solve_poisson(Ns(idx), f) - solve_u_true(Ns(idx), u)));
end

% Take logarithm of h and errors, avoiding log(0) issues
log_h = log(hs);
log_errors = log(errors);

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


% figure;  % Open a new figure
% loglog(hs, errors, 'b-', 'LineWidth', 2);  % 'b-' specifies a blue line
% % loglog(hs, errors, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k', 'MarkerSize', 8); % 'o' specifies circular markers
% grid on;  % Turn on the grid for better visibility
% xlabel('\(h\)','Interpreter', 'latex' )
% ylabel('\(\|u - u_h\|_{\infty}\)', 'Interpreter', 'latex')
% title('Error convergence plot')
% 
% 
% saveas(gcf, 'error_loglog.png')
% 






