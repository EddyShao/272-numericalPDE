D = 1e-3;
b_vec = [1 ; 0];
% f = @(x) D * (12 * (x(1) + x(2)) * sin(x(1) + 2 * x(2)) ...
%                   + 5 * (x(1) + x(2))^2 * cos(x(1) + 2 * x(2)) ...
%                   - 4 * cos(x(1) + 2 * x(2))) ...
%            + 2 * (x(1) + x(2)) * cos(x(1) + 2*x(2)) ...
%            - 2 * (x(1) + x(2))^2 *sin(x(1) + 2 * x(2));



u = @(x) (x(1) + x(2))^2 * cos(x(1) + 2 * x(2));

% g = u;

grad_u = @(x) [cos(x(1) + 2*x(2))*2*(x(1) + x(2)) - 1 * sin(x(1) + 2*x(2))*(x(1) + x(2))^2;
               cos(x(1) + 2*x(2))*2*(x(1) + x(2)) - 2 * sin(x(1) + 2*x(2))*(x(1) + x(2))^2];

% f = @(x) D * (12 * (x(1) + x(2)) * sin(x(1) + 2 * x(2)) ...
%                   + 5 * (x(1) + x(2))^2 * cos(x(1) + 2 * x(2)) ...
%                   - 4 * cos(x(1) + 2 * x(2))) + dot(b_vec, grad_u(x));

g = @(x) 0;
f = @(x) 1;


n_values = [4, 8, 16, 32, 64, 128];
L2_errors = zeros(size(n_values));
H1_errors = zeros(size(n_values));
h_values = 1 ./ n_values;  % Corresponding h values

% Loop over each n value and compute errors
for idx = 1:length(n_values)
    n = n_values(idx);
    
    % Solve the Poisson equation for the current n
    [U_h, node, element] = solve_conv_diff_2d_fem(0, 1, n, f, g, D, b_vec);
    % Plot the exact solution u
    figure;
    trisurf(element, node(:,1), node(:,2), U_h', 'EdgeColor', 'none');
    title(sprintf('Numerical Solution $u_h$ where $n=%02d$', n), 'Interpreter', 'latex');
    xlabel('$x$', 'Interpreter', 'latex');
    ylabel('$y$', 'Interpreter', 'latex');
    view(2); % Set view to 2D
    colorbar;
    axis equal;
    colormap(jet); % Set color map for better contrast
    shading interp; % Smooth color transitions
    filename = sprintf('3b_2_numerical_%02d.png', n); % Generate the filename with n
    saveas(gcf, filename);

end





% [U_h, node, element] = solve_conv_diff_2d_fem(0, 1, 64, f, g, D, b_vec);
% 
% u_exact_vals = arrayfun(@(i) u(node(i, :)), 1:size(node, 1));
% 
% figure;
% trisurf(element, node(:,1), node(:,2), u_exact_vals, 'EdgeColor', 'none');
% title('Exact Solution $u$', 'Interpreter', 'latex');
% xlabel('$x$', 'Interpreter', 'latex');
% ylabel('$y$', 'Interpreter', 'latex');
% view(2); % Set view to 2D
% colorbar;
% axis equal;
% colormap(jet);
% shading interp;
% saveas(gcf, '3b_2_exact_64.png');


