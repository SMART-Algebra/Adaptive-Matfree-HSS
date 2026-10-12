function [Y1, b1, Omega1, original_x, t] = test_system_toeplitz_1D(n, pt_leaf, k_A)
    % testToeplitzMatVec:
    %    Builds (conceptually) the 1-D Toeplitz matrix A where
    %        A(r, c) = 1 / (|r-c| * h)   for r~=c,
    %        A(r, r) = 2,
    %    but avoids forming it explicitly.
    %
    %    Then computes:
    %       Y = A * Omega
    %       b = A * original_x
    %
    % Inputs:
    %    n       : number of points (size of the matrix)
    %    pt_leaf : number of points per leaf (extra columns for Omega_i)
    %    k_A     : chosen rank (extra columns for Omega_i)
    %
    % Outputs:
    %    Y                  : n x max(pt_leaf + k_A, 3 * k_A) matrix
    %    b                  : n x 1 vector
    %    Omega              : random n x max(pt_leaf + k_A, 3 * k_A) test matrix
    %    original_x         : sinusoidal test vector
    %    t                  : 1-D FFT of the circulant embedding (for reuse)

    rng(0);

    h = 1 / n;
    x = (h : h : 1).';  % column vector of length n

    % t(1) = 2 (diagonal is 2), t(k) = 1/((k-1)*h) for k=2..n
    t = zeros(n,1);
    t(1) = 2; % diagonal entry
    for k = 2:n
        t(k) = (1/n) * (1 / ((k - 1) * h));
    end

    % --- Create random Omega_i and known solution original_x ---
    Omega1 = randn(n, max(pt_leaf + k_A, 3 * k_A));
    original_x = sin(pi * x);

    % --- Multiply Y1 = A * Omega1, column-by-column via FFT ---
    Y1 = zeros(n, size(Omega1,2));
    for col = 1:size(Omega1,2)
        Y1(:,col) = toeplitz_matVec_1D(Omega1(:,col), t);
    end

    % --- Multiply b1 = A * original_x via FFT ---
    b1 = toeplitz_matVec_1D(original_x, t);
end