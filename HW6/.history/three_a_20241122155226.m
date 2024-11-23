D = 1.;
b_vec = [1 ; 0];
% f = @(x) D * (12 * (x(1) + x(2)) * sin(x(1) + 2 * x(2)) ...
%                   + 5 * (x(1) + x(2))^2 * cos(x(1) + 2 * x(2)) ...
%                   - 4 * cos(x(1) + 2 * x(2))) ...
%            + 2 * (x(1) + x(2)) * cos(x(1) + 2*x(2)) ...
%            - 2 * (x(1) + x(2))^2 *sin(x(1) + 2 * x(2));



u = @(x) (x(1) + x(2))^2 * cos(x(1) + 2 * x(2));

g = u;

grad_u = @(x) [cos(x(1) + 2*x(2))*2*(x(1) + x(2)) - 1 * sin(x(1) + 2*x(2))*(x(1) + x(2))^2;
               cos(x(1) + 2*x(2))*2*(x(1) + x(2)) - 2 * sin(x(1) + 2*x(2))*(x(1) + x(2))^2];

f = @(x) D * (12 * (x(1) + x(2)) * sin(x(1) + 2 * x(2)) ...
                  + 5 * (x(1) + x(2))^2 * cos(x(1) + 2 * x(2)) ...
                  - 4 * cos(x(1) + 2 * x(2))) + dot(b_vec, grad_u(x));


n_values = [4, 8, 16, 32];
L2_errors = zeros(size(n_values));
H1_errors = zeros(size(n_values));
h_values = 1 ./ n_values;  % Corresponding h values

% Loop over each n value and compute errors
for idx = 1:length(n_values)
    n = n_values(idx);
    
    % Solve the Poisson equation for the current n
    [U_h, node, element] = solve_conv_diff_2d_fem(0, 1, n, f, g, D, b_vec);
    
    % Initialize error accumulators
    L_2_error = 0;
    H_1_error = 0;
    
    % Loop through each element
    for k = 1:size(element, 1)
        % Get the vertices of the k-th element
        node_ind = element(k, :);
        v_1 = node(node_ind(1), :);
        v_2 = node(node_ind(2), :);
        v_3 = node(node_ind(3), :);
        
        % Calculate area of the triangle element
        B = [(v_2 - v_1)', (v_3 - v_1)'];
        area = abs(det(B)) / 2;
        grad_phi = [-1, 1, 0; -1, 0, 1];
        B_T_inverse_grad_phi = B' \ grad_phi;
        
        % Calculate the approximate solution and exact solution at midpoints
        U_h_vals = [(U_h(node_ind(2)) + U_h(node_ind(3))) / 2, ...
                    (U_h(node_ind(3)) + U_h(node_ind(1))) / 2, ...
                    (U_h(node_ind(1)) + U_h(node_ind(2))) / 2];
        grad_U_h_vals = B_T_inverse_grad_phi * U_h(node_ind);
        grad_U_h_vals = repmat(grad_U_h_vals, 1, 3);  % Repeat for each midpoint

        % Midpoints of the edges
        midpoints = [(v_2 + v_3) / 2; (v_3 + v_1) / 2; (v_1 + v_2) / 2];
        u_vals = arrayfun(@(i) u(midpoints(i, :)), 1:3);
        grad_u_vals = [grad_u(midpoints(1, :)), grad_u(midpoints(2, :)), grad_u(midpoints(3, :))];
        
        % Accumulate L2 and H1 errors
        L_2_error = L_2_error + area * sum((U_h_vals - u_vals).^2) / 3;
        H_1_error = H_1_error + area * sum((grad_U_h_vals - grad_u_vals).^2, 'all') / 3;
    end
    
    % Finalize errors for this value of n
    L2_errors(idx) = sqrt(L_2_error);
    H1_errors(idx) = sqrt(H_1_error);
    
    % Print results
    fprintf('n = %d, h = %.4f, L2 norm error = %.6f, H1 norm error = %.6f\n', ...
            n, h_values(idx), L2_errors(idx), H1_errors(idx));
end

% Plot results in a log-log plot
figure;
loglog(h_values, L2_errors, '-o', 'DisplayName', 'L2 Norm Error');
hold on;
loglog(h_values, H1_errors, '-o', 'DisplayName', 'H1 Norm Error');
xlabel('h = 1/n');
ylabel('Error');
legend;
title('L2 and H1 Norm Errors vs. h');
grid on;
hold off;
saveas(gcf, '3a_error_plot.png');



[U_h, node, element] = solve_conv_diff_2d_fem(0, 1, 64, f, g, D, b_vec);

u_exact_vals = arrayfun(@(i) u(node(i, :)), 1:size(node, 1));
% Plot the exact solution u
figure;
trisurf(element, node(:,1), node(:,2), u_exact_vals, 'EdgeColor', 'none');
title('Exact Solution $u$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
view(2); % Set view to 2D
colorbar;
axis equal;
colormap(jet);
shading interp;
saveas(gcf, '3a_exact_64.png');


figure;
trisurf(element, node(:,1), node(:,2), U_h', 'EdgeColor', 'none');
title('Numerical Solution $u_h$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
view(2); % Set view to 2D
colorbar;
axis equal;
colormap(jet); % Set color map for better contrast
shading interp; % Smooth color transitions
saveas(gcf, '3a_numerical_64.png');

% generte a 3D plot
figure;
trisurf(element, node(:,1), node(:,2), U_h', 'EdgeColor', 'none');
title('Numerical Solution $u_h$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
view(3); % Set view to 3D
colorbar;
axis equal;
colormap(jet); % Set color map for better contrast
shading interp; % Smooth color transitions
saveas(gcf, '3a_numerical_64_3D.png');
