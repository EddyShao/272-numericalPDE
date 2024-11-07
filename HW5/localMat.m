
function [A_K, F_K] = generate_A_K(v_1, v_2, v_3, f)
    B = [v_2- v_1; v_3 - v_1];
    area = abs(det(B))/2;

    A_K = zeros(3, 3);
    grad_phi_1 = [-1; -1];
    grad_phi_2 = [1; 0];
    grad_phi_3 = [0; 1];
    grad_phi = [grad_phi_1, grad_phi_2, grad_phi_3];
    for i=1:3
        for j=1:3
            integrand = dot(B' \ grad_phi(:, i), B' \ grad_phi(:, j));
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
        mask(ind) = 0;
        F_K(ind) = (1/6) * area * dot(mask, f_value);
    end 

    
end


f = @(x_1, x_2) 1;

generate_A_K([0, 0], [0.5, 0.5], [0, 1], f)




    