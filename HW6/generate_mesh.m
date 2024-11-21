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
