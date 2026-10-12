function Z = apply_householder(W, C, X, mode)
    % Applies implicit Q using WY representation: Q = I - W * C'
    %
    % Inputs:
    %   W, C: The matrices from compute_householder
    %   X: Target matrix
    %   mode: 'normal' (Q*X) or 'transpose' (Q'*X)

    if strcmp(mode, 'transpose')
        % Q' = (I - W * C')' = I - C * W'
        % Q'X = X - C * (W' * X)
        
        Proj = W' * X;
        Z = X - C * Proj;
        
    else
        % Q = I - W * C'
        % QX = X - W * (C' * X)
        
        Proj = C' * X;
        Z = X - W * Proj;
    end
end