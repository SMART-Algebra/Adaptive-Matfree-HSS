function [t_solve, residual, solver_name, total_iters, true_relres, ...
        total_iters_np, solver_flag, unpreconditioned_flag] = ...
    solve_and_krylov(node, b, mv, num_levels, kernel_type, spd_kernel_list, symmetric_kernel_list, dim, compute_unpreconditioned)
%SOLVE_AND_KRYLOV ... COMPUTE_UNPRECONDITIONED (default true) controls the
%   unpreconditioned Krylov baseline. That baseline is a property of A
%   alone and does not depend on the factorization, so once it is published
%   there is no reason to pay for it again; it costs up to 10^4
%   unpreconditioned CG iterations per configuration. It is never run for
%   the Poisson-Schur kernels in any case.

    if nargin < 9 || isempty(compute_unpreconditioned)
        compute_unpreconditioned = true;
    end

    t_solve_time = tic;
    x_recovered  = linear_solve(node, b, num_levels, kernel_type, dim);
    t_solve      = round(toc(t_solve_time), 2);
    
    b_recovered = mv(x_recovered);
    residual    = norm(b - b_recovered) / norm(b);
    
    % Krylov with preconditioner Mfun
    Mfun = @(x) linear_solve(node, x, num_levels, kernel_type, dim);
    n = length(b);

    % One tolerance for every kernel and for both the preconditioned and
    % the unpreconditioned count.
    tol = 1e-12;

    maxit = 1000;
    restart = []; 
    b_norm = norm(Mfun(b)); % to normalize residuals for GMRES display

    if ismember(kernel_type, spd_kernel_list)
        solver_name = 'PCG';
        [Xg, flag, relres, iter, resvec] = pcg(mv, b, tol, maxit, Mfun);
        
        total_iters = iter;   
    else
        solver_name = 'GMRES';
        [Xg, flag, ~, iter, resvec] = gmres(mv, b, restart, tol, maxit, Mfun);
        
        if isempty(restart)
            total_iters = iter(2);
        else
            total_iters = (iter(1) - 1) * restart + iter(2);
        end
    end

    fprintf('%s residual history:\n', solver_name);
        for k = 1:length(resvec)
            % resvec(1) is ||r0||, so iteration count is k-1
            relres_k = resvec(k) / b_norm;
            fprintf('  iter %4d: relres = %.3e\n', k-1, relres_k);
        end
    
    solver_flag = flag;
    if flag == 3
        % Stagnation: the preconditioned residual reached the floor that
        % double precision allows for this factorization, within a factor
        % of a few above the tolerance, and MATLAB kept iterating until it
        % noticed. Its count then includes the stalled iterations (the
        % cavity problem at N=512^2, eps=1e-9: 1.45e-12 after 2 iterations,
        % unchanged through iteration 6, count 6). Report the iteration at
        % which the final residual was first reached; the flag and the
        % reached residual are stored alongside, so the table can mark it.
        % "Reached" means within one percent of the final residual: at
        % N=1024^2, eps=1e-9 the residual was 6.169e-12 at iteration 2 and
        % 6.168e-12 from iteration 3 on, and a tighter test reported 3.
        raw_iters = total_iters;
        floor_k = find(resvec <= min(resvec) * 1.01, 1, 'first');
        total_iters = floor_k - 1;
        fprintf(['%s stagnated (flag 3) at relres %.3e: MATLAB counted %d ' ...
            'iterations, the final residual was reached at iteration %d, ' ...
            'which is reported.\n'], solver_name, resvec(end) / b_norm, ...
            raw_iters, total_iters);
    end
    fprintf('Solver finished. Flag = %d. Total iters = %d.\n', ...
        solver_flag, total_iters);

    total_iters_np = NaN;
    unpreconditioned_flag = NaN;
    
    if ~compute_unpreconditioned
        fprintf('Unpreconditioned baseline skipped; reusing published counts.\n');
    elseif strcmpi(kernel_type, 'helmholtz') || strcmpi(kernel_type, 'helmholtz_schur')
        % No restart, matching the preconditioned solve above: a restarted
        % GMRES discards its Krylov history at every cycle, so an
        % artificial restart in the unpreconditioned baseline would
        % inflate its count for a reason unrelated to the preconditioner.
        % Reuses TOL and MAXIT from the preconditioned solve above, so the
        % two counts are comparable by construction. Both symmetric
        % indefinite kernels (the Lippmann-Schwinger operator, 'helmholtz',
        % and the 2D Helmholtz strip Schur complement, 'helmholtz_schur')
        % take this branch.
        %
        % The unpreconditioned count can be far larger than MAXIT (the
        % Helmholtz strip needs 5100 at N=16384, the cavity potential more
        % than 715 at N=1024^2), so it gets its own cap: 10000, reduced so
        % that the unrestarted Krylov basis (N x maxit complex doubles)
        % fits a memory budget. HSS_UNPRECOND_MEM_BYTES sets the budget;
        % the default 12e9 is laptop-safe, and the Stampede3 jobs export a
        % larger one. A capped, unconverged run is reported with flag 1
        % and its count is then a lower bound.
        mem_bytes = str2double(getenv('HSS_UNPRECOND_MEM_BYTES'));
        if ~isfinite(mem_bytes) || mem_bytes <= 0, mem_bytes = 12e9; end
        maxit_np = min(10000, floor(mem_bytes / (n * 16)));
        [~, flag_np, ~, iter_np, resvec_np] = gmres(mv, b, [], tol, maxit_np);
        total_iters_np = iter_np(2);
        if maxit_np < 10000
            fprintf('Unpreconditioned GMRES maxit capped at %d by the %.0f GB memory budget\n', ...
                maxit_np, mem_bytes / 1e9);
        end

        unpreconditioned_flag = flag_np;
        fprintf('Unpreconditioned (GMRES): flag = %d, iters = %d\n', ...
            unpreconditioned_flag, total_iters_np);

    elseif strcmpi(kernel_type, 'matern')
        tol_np   = 1e-12;
        maxit_np = 10000;

        [~, flag_np, ~, iter_np, resvec_np] = pcg(mv, b, tol_np, maxit_np);
        
        total_iters_np = iter_np;

        unpreconditioned_flag = flag_np;
        fprintf('Unpreconditioned (PCG): flag = %d, iters = %d\n', ...
            unpreconditioned_flag, total_iters_np);
    end
    
    b_rec       = mv(Xg);
    true_relres = norm(b - b_rec) / norm(b);
end
