syms x1 x2

% Define the function u
u = (x1 + x2)^2 * cos(x1 + 2 * x2);

% Compute the gradient of u with respect to x1 and x2
grad_u = gradient(u, [x1, x2]);

% Display the result
disp('Gradient of u:');
disp(grad_u);