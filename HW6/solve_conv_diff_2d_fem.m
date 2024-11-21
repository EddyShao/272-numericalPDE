function [U_h, node, element] = solve_conv_diff_2d_fem(a, b, n, f, g, D, b_vec)
    [node, element, bdNode] = generate_mesh(n, a, b);
    A = zeros(size(node, 1));
    F = zeros(size(node, 1), 1);
    for ind=1:size(element, 1)
        node_ind = element(ind, :);
        [A_K, F_K] = generate_A_F_K(node(node_ind(1), :), node(node_ind(2), :), node(node_ind(3), :), f, D, b_vec);
        
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






