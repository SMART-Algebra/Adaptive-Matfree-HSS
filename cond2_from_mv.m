function [kappa2, stats] = cond2_from_mv(n, mv, mvT, node, num_levels, kernel_type, dim)
%  Estimate the 2-norm condition number kappa2(A)
%  using only matrix–vector products (A*x and A^*x).

    % --- Fixed parameters (tune if needed) ---
    num_power_iters    = 50;     % for largest singular value
    tol                = 1e-8;
    num_inverse_iters  = 50;     % for smallest singular value
    maxit_cg           = 1000;

    % Operator for B = A^*A (HPD)
    apply_B = @(x) mvT(mv(x));

    % ==============================================================
    % Estimate largest eigenvalue of B using power method
    % ==============================================================
    v = randn(n,1);
    v = v / norm(v);
    lambda = NaN;
    total_power_iters = 0;

    for iter = 1:num_power_iters
        total_power_iters = total_power_iters + 1;
        w = apply_B(v);
        w_norm = norm(w);
        if w_norm == 0
            error('Power method failed: apply_B(v) == 0.');
        end

        lambda = real(v' * w);   % Rayleigh quotient
        if norm(w - lambda * v) <= tol * w_norm
            break
        end

        v = w / w_norm;
    end
    lambda_max = lambda;

    % ==============================================================
    % Estimate smallest eigenvalue using inverse power method
    %     Solve (A^*A) y = v using gmres each step
    % ==============================================================

    v = randn(n,1);
    v = v / norm(v);
    lambda = NaN;
    total_inverse_iters = 0;
    M_A  = @(x) linear_solve(node, x, num_levels, kernel_type, dim);
    Pinv = @(x) M_A(M_A(x));

    for iter = 1:num_inverse_iters
        total_inverse_iters = total_inverse_iters + 1;
        
        warning_state = warning;
        warning('off', 'all');
        [w, cg_flag, ~, cg_iters] = gmres(apply_B, v, n, 1e-6, maxit_cg, Pinv);
        % [w, cg_flag, ~, cg_iters] = pcg(apply_B, v, 1e-6, maxit_cg, Pinv);
        warning(warning_state);

        w_norm = norm(w);
        if w_norm == 0
            error('Power method failed: B \ v == 0.');
        end
        lambda = 1/real(v' * w);   % Rayleigh quotient
        r = apply_B(v);
        if norm(r - lambda * v) <= tol * norm(r)
            break
        end

        v = w / w_norm;
    end
    lambda_min = lambda;

    % ==============================================================
    % Compute singular values and condition number
    % ==============================================================
    sigma_max = sqrt(lambda_max);
    sigma_min = sqrt(lambda_min);
    if sigma_min == 0
        kappa2 = Inf;
    else
        kappa2 = sigma_max / sigma_min;
    end

    % ==============================================================
    % Return info
    % ==============================================================
    if nargout > 1
        stats = struct( ...
            'lambda_max', lambda_max, ...
            'lambda_min', lambda_min, ...
            'sigma_max',  sigma_max, ...
            'sigma_min',  sigma_min, ...
            'total_power_iters',   total_power_iters, ...
            'total_inverse_iters', total_inverse_iters, ...
            'cg_last_flag', cg_flag, ...
            'cg_last_iters', cg_iters);
    end
end