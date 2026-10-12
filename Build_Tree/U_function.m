function [W_i, C_i] = U_function(k, Y_i, Omega_i) 
    % UFunction computes U matrices in Householder form (W, C)
    % Inputs:
    %   k              - Estimated maximum rank
    %   Y_i            - Y block for the current node
    %   Omega_i        - Omega block for the current node
    %
    % Outputs:
    %   W_i, C_i       - Householder vectors representing U_i
    %                    such that U_i (implicit) = I - W_i * C_i'


    % hat_Q_i is null space basis: last k columns of the full Q of
    % qr(Omega_i'), without forming that Q when the MEX is built
    hat_Q_i = null_space_basis(Omega_i, k);  % (block_size + k) x k

    Y_i_hat = Y_i * hat_Q_i;  % block_size x k

    [W_i, C_i] = compute_householder(Y_i_hat);
end