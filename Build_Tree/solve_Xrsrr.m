function [X_rs_i, X_rr_i] = solve_Xrsrr(Y_i_tilde, Omega_i_tilde, nodeIndices_i, kernel_type, symmetric_kernel_list, spd_kernel_list, level)
    % solveXrsrr: Compute X_rs, X_rr by unregularized least squares.
    %   The local systems are overdetermined with condition numbers of order
    %   ten, so a regularizer buys nothing and the backslash QR is used
    %   directly. A rank-deficient system now warns instead of being
    %   silently damped.

    if nargin < 7, level = NaN; end
    
    num_skele = length(nodeIndices_i.skeleton);

    % Partition into s and r
    Y_tilde_i_r = Y_i_tilde(num_skele+1:end, :);
    Omega_tilde_i_s = Omega_i_tilde(1:num_skele, :);
    Omega_tilde_i_r = Omega_i_tilde(num_skele+1:end, :);

    % =========================================================
    % Solve X_rr
    % =========================================================
    % Solves [X_rs, X_rr] * W = Y_r
    % Transposed: W' * [X_rs, X_rr]' = Y_r'

    W_transpose = [Omega_tilde_i_s; Omega_tilde_i_r]';
    RHS_X_rr = Y_tilde_i_r';

    X_rs_rr_i_trans = W_transpose \ RHS_X_rr;
    
    record_ls_diagnostic(W_transpose, RHS_X_rr, X_rs_rr_i_trans, level);

    X_rs_rr_i = X_rs_rr_i_trans';
    X_rr_raw = X_rs_rr_i(:, num_skele+1:end);

    % =========================================================
    % Enforce symmetry/spd
    % =========================================================
    if ismember(kernel_type, spd_kernel_list)    
        X_rr_i = 0.5 * (X_rr_raw + X_rr_raw.');
        [~, p] = chol(X_rr_i);
        if p > 0
            matrix_norm = norm(X_rr_i, 'fro');
            shift = 1e-12 * matrix_norm;
            if shift == 0, shift = 1e-12; end
            while p > 0
                X_rr_i = X_rr_i + shift * eye(size(X_rr_i));
                [~, p] = chol(X_rr_i);
                shift = shift * 2;
            end
        end
    else
        if ismember(kernel_type, symmetric_kernel_list)
            X_rr_i = 0.5 * (X_rr_raw + X_rr_raw.');
        else
            X_rr_i = X_rr_raw;
        end
    end

    % =========================================================
    % Solve X_rs
    % =========================================================
    
    RHS_for_rs = Y_tilde_i_r - X_rr_i * Omega_tilde_i_r;
    
    % Solve Omega_s' * X_rs' = RHS'
    A_X_rs = Omega_tilde_i_s';
    b_X_rs = RHS_for_rs';

    X_rs_trans = A_X_rs \ b_X_rs;
    
    X_rs_i = X_rs_trans';

    record_xrr_diagnostic(X_rr_i, X_rs_i, level);
end
function record_ls_diagnostic(W_transpose, RHS, SOL, level)
%RECORD_LS_DIAGNOSTIC Log the conditioning of the local test block and the
%   least-squares residual. Inert unless LS_DIAG.enabled is set, so the
%   production path pays nothing.
    global LS_DIAG %#ok<GVMIS>
    if isempty(LS_DIAG) || ~isstruct(LS_DIAG) || ~LS_DIAG.enabled
        return;
    end
    denom = norm(RHS, 'fro');
    % A node whose transformed redundant sample is negligible carries no
    % information about the model error; record the scale so those nodes can
    % be excluded rather than silently reported as a zero residual.
    if denom == 0
        resid = NaN;
    else
        resid = norm(RHS - W_transpose * SOL, 'fro') / denom;
    end
    % Isotropy of the transformed test block: the Hutchinson identity behind
    % the stopping test needs E[Omega Omega'] proportional to the identity.
    % For an m-by-p Gaussian, (1/p) Omega Omega' concentrates on I, so the
    % deviation below measures how far the propagated tests have drifted from
    % that assumption. A Gaussian of the same shape gives a nonzero baseline,
    % which the driver computes for comparison.
    Om = W_transpose';                       % m-by-p transformed test block
    p_cols = size(Om, 2);
    iso_dev = norm((Om * Om') / p_cols - eye(size(Om, 1)), 2);

    sv = svd(W_transpose);
    smax = max(sv); smin = min(sv);
    if smin > 0, kappa = smax / smin; else, kappa = Inf; end
    % One row per node: [level, rows, cols, kappa, sigma_min, residual].
    % size(RHS,2) is the number of redundant indices at this node. When it is
    % zero the node is all skeleton, nothing is eliminated, and the solve is
    % vacuous -- such nodes must not be pooled with the real ones.
    LS_DIAG.rows(end+1, :) = [level, size(W_transpose, 1), size(W_transpose, 2), ...
        kappa, smin, resid, denom, size(RHS, 2), iso_dev];
end

function record_xrr_diagnostic(X_rr, X_rs, level)
%RECORD_XRR_DIAGNOSTIC Conditioning of the compressed diagonal block and the
%   amplification it causes. Inert unless XRR_DIAG.enabled is set.
%
%   The elimination forms X_sr * inv(X_rr) and applies it to the child's
%   samples, so a child's compression error reaches its parent multiplied by
%   the norm of that product. If X_rr is ill conditioned the factor is large,
%   which is the natural explanation for accuracy that degrades with depth.
%   One row per node: [level, r, cond(X_rr), sigma_min, sigma_max,
%   ||X_rs||, ||X_sr inv(X_rr)||].
    global XRR_DIAG %#ok<GVMIS>
    if isempty(XRR_DIAG) || ~isstruct(XRR_DIAG) || ~XRR_DIAG.enabled
        return;
    end
    if isempty(X_rr) || size(X_rr, 1) == 0
        return;
    end
    sv = svd(X_rr);
    smax = max(sv);
    smin = min(sv);
    if smin > 0, kappa = smax / smin; else, kappa = Inf; end
    amp = NaN;
    if ~isempty(X_rs) && size(X_rs, 2) > 0
        amp = norm(X_rs.' / X_rr, 2);
    end
    XRR_DIAG.rows(end+1, :) = [level, size(X_rr, 1), kappa, smin, smax, ...
        norm(X_rs, 2), amp];
end
