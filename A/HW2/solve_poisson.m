a = -1;
b = exp(2)*(cos(1) + 2*sin(1));

f = @(x) -exp(x.^2+1).*(4*x.*cos(x) + sin(x) + 4*x.^2.*sin(x));

u = @(x) sin(x).*exp(x.^2+1) -1;

% solve 1D poisson equation with Nuemann Boundary condition
% with ghost grid point
Ns = 10:10:3000;  % Calculate the inverse and convert to integers
hs = 1 ./ Ns;
errors = zeros(length(Ns));
for idx = 1:length(Ns)
    errors(idx) = max(abs(solve_poisson(a, b, Ns(idx), f) - solve_u_true(Ns(idx), u)));
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


figure;  % Open a new figure
loglog(hs, errors, 'b-', 'LineWidth', 2);  % 'b-' specifies a blue line
hold on;  % Hold the current plot to overlay markers
loglog(hs, errors, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k', 'MarkerSize', 8); % 'o' specifies circular markers
hold off;  % Release the hold on the current plot
grid on;  % Turn on the grid for better visibility
ax = gca;  % Get the current axes
ax.Toolbar.Visible = 'off';  % Turn off the axes toolbar
xlabel('\(h\)','Interpreter', 'latex' )
ylabel('\(\|u - u_h\|_{\infty}\)', 'Interpreter', 'latex')
title('Error convergence plot')


saveas(gcf, 'error_loglog_symmetric.png')

function A = poisson_nuemann_mat_1d(N)
    e = ones(N, 1);  % Block of ones for the neighboring diagonal
    A = spdiags([-e 2*e -e], -1:1, N, N);  % 1D Poisson matrix for each block
    A(N, N) = 1;
end



function u_h = solve_poisson(a, b, N, f)
    h = (1 - 0)/ N;
    x = linspace(0, 1, N+1);
    f_value = f(x(2:N+1))';
    f_value(1, 1) = f_value(1, 1) + a/h^2;
    f_value(end, 1) = f_value(end, 1)/2 + b/h;
    A = (1/h^2) * poisson_nuemann_mat_1d(N);
    u_h = A \ f_value;
end

function u_true = solve_u_true(N, u)
    x = linspace(0, 1, N+1);
    u_true = u(x(2:N+1))';
end

% u_h = solve_poisson(a, b, 100000, f);
% u_true = solve_u_true(100000, u);
% max(abs(u_h - u_true))



