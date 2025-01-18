f = @(x) 12 * (x(1) + x(2)) * sin(x(1) + 2 * x(2)) ...
                  + 5 * (x(1) + x(2))^2 * cos(x(1) + 2 * x(2)) ...
                  - 4 * cos(x(1) + 2 * x(2));

u = @(x) (x(1) + x(2))^2 * cos(x(1) + 2 * x(2));

g = u;

grad_u = @(x) [cos(x(1) + 2*x(2))*2*(x(1) + x(2)) - 1 * sin(x(1) + 2*x(2))*(x(1) + x(2))^2;
               cos(x(1) + 2*x(2))*2*(x(1) + x(2)) - 2 * sin(x(1) + 2*x(2))*(x(1) + x(2))^2];

function [node, element, bdNode] = generate_mesh(n, a, b)
    % Number of nodes per side
    numNodes = n + 1;
    
    % Step size for uniform grid
    h = (b - a) / n;
    
    % Generate grid points
    [y, x] = meshgrid(linspace(a, b, numNodes));
    node = [x(:), y(:)];
    
    % Initialize element matrix
    element = zeros(2 * n^2, 3);

    % assemble the element matrix
    % l:left, r:right, b:bottom, t:top
    for row = 1:n
        for col = 1:n
            lb = (row - 1)*(n + 1) + col;
            rb = lb + 1; 
            lt = lb + (n+1);
            rt = lt + 1;
            ind_1 = (row - 1)*n + col;
            ind_2 = ind_1 + n^2;
            element(ind_1, :) = [lb, lt, rt];
            element(ind_2, :) = [lb, rb, rt];
        end
    end 
    
    % Initialize boundary node vector
    bdNode = zeros(size(node, 1), 1);
    
    % Identify boundary nodes
    tol = h*1e-3;
    bdNode(abs(x(:)- a)<tol |abs(x(:)- b)<tol | abs(y(:)- a)<tol | abs(y(:)- b)<tol) = 1;
end


%%%%%%%%%%%%%%%% Test for probelm 1 %%%%%%%%%%%%%%%%
[node, element, bdNode] = generate_mesh(5, 0, 1);
% Print the 7th row of the node matrix
fprintf('7th row of the node matrix:\n');
disp(node(7, :));

% Print the 8th row of the element matrix
fprintf('8th row of the element matrix:\n');
disp(element(8, :));

% Print the 12th component of the bdNode vector
fprintf('12th component of the bdNode vector:\n');
disp(bdNode(12));
%%%%%%%%%%%%%%%% Test for probelm 1 %%%%%%%%%%%%%%%%


function [A_K, F_K] = generate_A_F_K(v_1, v_2, v_3, f)
    B = [(v_2- v_1)', (v_3 - v_1)'];
    area = abs(det(B))/2;

    A_K = zeros(3, 3);
    % grad_phi_1 = [-1; -1];
    % grad_phi_2 = [1; 0];
    % grad_phi_3 = [0; 1];
    % grad_phi = [grad_phi_1, grad_phi_2, grad_phi_3];
    grad_phi = [-1, 1, 0; -1, 0, 1];
    B_T_inverse_grad_phi = B' \ grad_phi;
    for i=1:3
        for j=1:3
            integrand = dot(B_T_inverse_grad_phi(:, i), B_T_inverse_grad_phi(:, j));
            A_K(i, j) = integrand*area;
        end
    end

    F_K = zeros(3, 1);
    m_1 = (v_2 + v_3) / 2;
    m_2 = (v_1 + v_3) / 2;
    m_3 = (v_1 + v_2) / 2;
    f_value = [f(m_1), f(m_2), f(m_3)];
    for ind=1:3
        mask = ones(1, 3);
        mask(1, ind) = 0;
        F_K(ind) = (1/6) * area * dot(mask, f_value);
    end     
end


%%%%%%%%%%%%%%%% Test for probelm 2 %%%%%%%%%%%%%%%%

% Define the vertices
v_1 = [0, 0];
v_2 = [0.5, 0.5];
v_3 = [0, 1];

% Define the source function f
% f = @(x) 1;  % Example constant source term, modify as needed

% Generate the local stiffness matrix A_K and local load vector F_K
[A_K, F_K] = generate_A_F_K(v_1, v_2, v_3, f);

% Display the results
fprintf('Local stiffness matrix A_K:\n');
disp(A_K);

fprintf('Local load vector F_K:\n');
disp(F_K);

%%%%%%%%%%%%%%%% Test for probelm 2 %%%%%%%%%%%%%%%%


function [U_h, node, element] = solve_poisson_2d_fem(a, b, n, f, g)
    [node, element, bdNode] = generate_mesh(n, a, b);
    A = zeros(size(node, 1));
    F = zeros(size(node, 1), 1);
    for ind=1:size(element, 1)
        node_ind = element(ind, :);
        [A_K, F_K] = generate_A_F_K(node(node_ind(1), :), node(node_ind(2), :), node(node_ind(3), :), f);
        
        for i = 1:3
            for j = 1:3
                A(node_ind(i), node_ind(j)) = A(node_ind(i), node_ind(j)) + A_K(i, j);
                
                % Display the current state of A
               
            end
        end
        
        for i=1:3
            F(node_ind(i)) = F(node_ind(i)) + F_K(i);
        end

    end

    U_0 = arrayfun(@(i) g(node(i, :)), 1:size(node, 1))';
    U_0 = U_0 .* bdNode;
    U_h = U_0;
    F = F - A*U_0;
    indices = (bdNode == 0);
    A_reduced = A(indices, indices);
    F_reduced = F(indices); 
    U_I = A_reduced \ F_reduced;
    U_h(indices) = U_I;

end






n_values = [4, 8, 16, 32];
L2_errors = zeros(size(n_values));
H1_errors = zeros(size(n_values));
h_values = 1 ./ n_values;  % Corresponding h values

% Loop over each n value and compute errors
for idx = 1:length(n_values)
    n = n_values(idx);
    
    % Solve the Poisson equation for the current n
    [U_h, node, element] = solve_poisson_2d_fem(0, 1, n, f, g);
    
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
saveas(gcf, 'error_plot.png');



[U_h, node, element] = solve_poisson_2d_fem(0, 1, 64, f, g);

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
saveas(gcf, 'exact_64.png');


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
saveas(gcf, 'numerical_64.png');