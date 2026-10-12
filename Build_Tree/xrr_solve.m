function Z = xrr_solve(X_rr, B, side)
%XRR_SOLVE Solve with the compressed diagonal block, optionally truncated.
%   Z = XRR_SOLVE(X_RR, B, 'left')  computes X_RR \ B.
%   Z = XRR_SOLVE(X_RR, B, 'right') computes B / X_RR.
%
%   By default this is exactly the backslash it replaces. Setting the global
%   XRR_TRUNC to a relative threshold tau > 0 instead discards the singular
%   directions of X_RR below tau * sigma_max and applies the pseudo-inverse
%   of what remains.
%
%   WHY. The elimination forms X_sr * inv(X_rr) and applies it to a child's
%   samples, so a child's compression error reaches its parent multiplied by
%   the norm of that product. Measured at eps = 1e-9 on 2026-09-14, the
%   condition number of X_rr climbs steeply toward the root: 1.9, 4.4, 23 for
%   2D Poisson at 128^2, but 12, 350, 3.6e5 for Matern (1.5, 0.1), and
%   1.6e6 at the root at 256^2. Within a kernel the delivered accuracy tracks
%   that number, growing as roughly its 0.8 to 1.6 power.
%
%   X_rr is itself the solution of a sampled least-squares problem, so its
%   smallest singular values are estimation noise rather than structure.
%   Inverting them amplifies that noise into the parent. Truncating below a
%   threshold tied to the requested tolerance discards exactly those
%   directions, at the price of an elimination that is no longer exact.
%
%   See also SOLVE_XRSRR.
    global XRR_TRUNC XRR_FACT %#ok<GVMIS>
    tau = 0;
    if ~isempty(XRR_TRUNC) && isnumeric(XRR_TRUNC) && isscalar(XRR_TRUNC)
        tau = XRR_TRUNC;
    end
    if tau <= 0 || isempty(X_rr) || isempty(B)
        % Backslash on a dense square matrix tries Cholesky when the matrix
        % is symmetric with a positive diagonal and otherwise factors with
        % partially pivoted LU; it does not use a symmetric indefinite
        % factorization. For the indefinite kernels X_rr is exactly
        % symmetric (SOLVE_XRSRR symmetrizes it) but not definite, so the
        % Cholesky attempt fails and the LU path is taken, ignoring the
        % symmetry. Setting the global XRR_FACT to 'ldl' uses the
        % Bunch-Kaufman factorization instead: half the arithmetic, and the
        % symmetry preserved. It does not change the accuracy, because the
        % amplification that costs the digits is norm(X_sr * inv(X_rr)), a
        % property of the two matrices rather than of how the inverse is
        % applied; both factorizations are backward stable, and the forward
        % error each incurs is smaller than the compression error by orders
        % of magnitude. Measured on the Helmholtz Schur complement of
        % Section 5.3, the delivered residuals agree to seven digits.
        use_ldl = ~isempty(XRR_FACT) && (ischar(XRR_FACT) || isstring(XRR_FACT)) ...
            && strcmpi(XRR_FACT, 'ldl') && ~isempty(X_rr) && issymmetric(X_rr);
        if use_ldl
            [L, D, P] = ldl(X_rr);
            if strcmpi(side, 'left')
                Z = P * (L' \ (D \ (L \ (P' * B))));
            else
                Z = ((P * (L' \ (D \ (L \ (P' * B'))))))';
            end
        elseif strcmpi(side, 'left')
            Z = X_rr \ B;
        else
            Z = B / X_rr;
        end
        return;
    end
    [Us, S, Vs] = svd(X_rr, 'econ');
    s = diag(S);
    if isempty(s) || s(1) == 0
        Z = zeros(size(B, 1), size(X_rr, 2));
        if strcmpi(side, 'left'), Z = zeros(size(X_rr, 2), size(B, 2)); end
        return;
    end
    keep = s > tau * s(1);
    si = zeros(size(s));
    si(keep) = 1 ./ s(keep);
    % X_rr = Us*S*Vs', so inv(X_rr) = Vs*inv(S)*Us'.
    if strcmpi(side, 'left')
        Z = Vs * (si .* (Us' * B));
    else
        Z = ((B * Vs) .* si.') * Us';
    end
end
