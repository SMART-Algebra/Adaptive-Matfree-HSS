function results = run_ls_conditioning_diagnostic(varargin)
%RUN_LS_CONDITIONING_DIAGNOSTIC Conditioning of the Section 3.2 least squares.
%   Records cond(Omega_tilde_i) and the relative least-squares residual at
%   every node of every level, for each test problem, and summarizes them by
%   level. Answers whether the recovery of the redundant diagonal blocks needs
%   regularization at all, and if so at which levels.
%
%   Name-value options:
%     'method'      nullspace update method (default 'householder')
%     'tolerance'   stopping tolerance epsilon (default 1e-6, the value the
%                   manuscript's conditioning table states)
%     'step_b'      sampling block size b: a scalar, or one value per
%                   problem in the list below (default 32)
%     'only'        indices into the problem list; [] runs all four
%     'pt_leaf'     override the per-problem leaf size
%     'num_levels'  override the per-problem tree depth
%     'rank_pruning' 'all' (default, the Section 5 protocol), 'leaf' or 'none'
%     'output_dir'  when nonempty, write three files into it:
%                   ls_conditioning_summary.csv, one row per problem and
%                   level with the columns of the manuscript table
%                   (equations, unknowns, rho, kappa measured, kappa_MP)
%                   plus the diagnostics printed here;
%                   ls_conditioning_nodes.csv, one row per node; and
%                   rank_search.csv, how the rank search exited at every
%                   level. Existing files are overwritten.
%
%   The manuscript's protocol uses b = 16 for the 1D problem and b = 32 for
%   the 2D problems, so the production call is
%     run_ls_conditioning_diagnostic('step_b', [16 32 32 32], ...
%         'tolerance', 1e-6, 'output_dir', dir)

    p = inputParser;
    p.addParameter('method', 'householder');
    p.addParameter('tolerance', 1e-6);
    p.addParameter('step_b', 32);
    p.addParameter('only', []);   % indices into the problem list; [] runs all
    p.addParameter('pt_leaf', []);% override the per-problem leaf size
    p.addParameter('num_levels', []);% override the per-problem tree depth
    p.addParameter('rank_pruning', 'all');
    p.addParameter('output_dir', '');
    p.parse(varargin{:});
    opt = p.Results;

    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));

    % The Matern case is (nu, ell) = (0.5, 0.5), the pair the block-size
    % study in the same appendix uses.
    problems = { ...
        struct('label','1D Poisson Schur',  'dim',1,'kernel','poisson_schur','kernel_params',struct(), 'num_levels',6,'pt_leaf',256), ...
        struct('label','2D Poisson Schur',  'dim',2,'kernel','poisson_schur','kernel_params',struct(), 'num_levels',4,'pt_leaf',256), ...
        struct('label','2D Matern',         'dim',2,'kernel','matern',       'kernel_params',struct('nu',0.5,'ell',0.5), 'num_levels',4,'pt_leaf',256), ...
        struct('label','Lippmann-Schwinger','dim',2,'kernel','helmholtz',    'kernel_params',struct('potential','cavity'), 'num_levels',4,'pt_leaf',256)};

    step_b = opt.step_b(:)';
    if isscalar(step_b)
        step_b = repmat(step_b, 1, numel(problems));
    elseif numel(step_b) ~= numel(problems)
        error(['step_b must be a scalar or have one entry per problem ' ...
            '(%d); got %d entries.'], numel(problems), numel(step_b));
    end
    if ~isempty(opt.only)
        only = opt.only(:)';
        if ~isnumeric(only) || any(only ~= round(only)) || any(only < 1) ...
                || any(only > numel(problems)) || numel(unique(only)) ~= numel(only)
            error(['only must be a list of distinct problem indices between ' ...
                '1 and %d; got %s.'], numel(problems), mat2str(opt.only));
        end
        problems = problems(only);
        step_b = step_b(only);
    end
    % Validate every selected problem before any of them runs, so that a leaf
    % size the 2D problems cannot take fails at once instead of after the 1D
    % problem has been factorized.
    for ip = 1:numel(problems)
        pr = problems{ip};
        if ~isempty(opt.pt_leaf), pr.pt_leaf = opt.pt_leaf; end
        if pr.dim == 2 && mod(sqrt(pr.pt_leaf), 1) ~= 0
            error('For dim = 2 the leaf size must be a perfect square; got %d (%s).', ...
                pr.pt_leaf, pr.label);
        end
    end

    global LS_DIAG RANK_DIAG %#ok<GVMIS>
    results = struct('problem', {}, 'table', {}, 'summary', {});
    summary_rows = table();
    node_rows = table();
    rank_rows = table();

    for ip = 1:numel(problems)
        pr = problems{ip};
        if ~isempty(opt.pt_leaf), pr.pt_leaf = opt.pt_leaf; end
        if ~isempty(opt.num_levels), pr.num_levels = opt.num_levels; end
        LS_DIAG   = struct('enabled', true, 'rows', zeros(0, 9));
        RANK_DIAG = struct('enabled', true, 'rows', zeros(0, 6));

        params = struct('dim', pr.dim, 'user_tolerance', opt.tolerance, ...
            'step_b', step_b(ip), 'num_levels', pr.num_levels, ...
            'pt_leaf', pr.pt_leaf, 'method', opt.method, ...
            'kernel_type', pr.kernel, 'kernel_params', pr.kernel_params, ...
            'warmup', false, ...
            'rank_pruning', opt.rank_pruning, 'compute_cond', false);
        try
            evalc('[~, run_stats] = main(params);');
            n_total = run_stats.n;
        catch err
            fprintf('\n=== %s: FAILED (%s) ===\n', pr.label, err.message);
            LS_DIAG.enabled = false;
            RANK_DIAG.enabled = false;
            continue;
        end
        rec = LS_DIAG.rows;
        rnk = RANK_DIAG.rows;
        LS_DIAG.enabled = false;
        RANK_DIAG.enabled = false;

        fprintf('\n=== %s (dim %d, N = %d, %d levels, leaf %d, b = %d, eps = %.0e, pruning %s, %d nodes solved) ===\n', ...
            pr.label, pr.dim, n_total, pr.num_levels, pr.pt_leaf, step_b(ip), ...
            opt.tolerance, opt.rank_pruning, size(rec, 1));
        fprintf('%7s %7s %8s %8s %6s %8s %8s %11s %7s %8s %11s %11s %10s\n', ...
            'level', 'nodes', 'eqns', 'unknowns', 'rho', 'kappa', 'kappa_MP', ...
            'max kappa', 'redund', 'vacuous', 'med resid', 'max resid', 'isotropy');
        fprintf('%s\n', repmat('-', 1, 122));

        lv = rec(:, 1)';
        tbl = [];
        for L = unique(lv(~isnan(lv)))
            m = lv == L;
            nr = rec(m, 2)'; nc = rec(m, 3)'; nred = rec(m, 8)';
            kap = rec(m, 4)'; smin = rec(m, 5)'; res = rec(m, 6)'; rhs = rec(m, 7)';
            iso = rec(m, 9)';
            % A vacuous node has no redundant indices, so its "solve" has no
            % unknowns and its condition number describes nothing. Report the
            % count, but keep those nodes out of every statistic.
            real_node = nred > 0;
            n_vacuous = nnz(~real_node);
            if any(real_node)
                nr = nr(real_node); nc = nc(real_node); kap = kap(real_node);
                smin = smin(real_node); res = res(real_node);
                rhs = rhs(real_node); iso = iso(real_node);
                nred_stat = nred(real_node);
            else
                nred_stat = nred;
            end
            % Exclude nodes whose right-hand side is numerically zero: their
            % relative residual is 0/0, not evidence that the model holds.
            live = isfinite(res) & rhs > eps * max(rhs);
            if ~any(live), live = isfinite(res); end
            res = res(live);
            if isempty(res), res = NaN; end
            % W_transpose is (equations) x (unknowns per column of X): the
            % solve is overdetermined only when equations > unknowns. The
            % manuscript's aspect ratio rho is equations over unknowns, and
            % kappa_MP is the Marchenko-Pastur condition number of a Gaussian
            % matrix of that aspect ratio, (sqrt(rho)+1)/(sqrt(rho)-1).
            eqns = median(nr); unknowns = median(nc);
            rho = eqns / unknowns;
            if rho > 1
                kappa_mp = (sqrt(rho) + 1) / (sqrt(rho) - 1);
            else
                kappa_mp = Inf;
            end
            fprintf('%7d %7d %8d %8d %6.2f %8.1f %8.1f %11.1f %7d %8d %11.2e %11.2e %10.3f\n', ...
                L, nnz(m), eqns, unknowns, rho, median(kap), kappa_mp, max(kap), ...
                median(nred_stat), n_vacuous, median(res), max(res), median(iso));
            tbl = [tbl; L, nnz(m), eqns, unknowns, rho, median(kap), kappa_mp, max(kap), ...
                median(nred_stat), n_vacuous, min(smin), median(res), max(res), median(iso)]; %#ok<AGROW>
            summary_rows = [summary_rows; table( ...
                string(pr.label), pr.dim, n_total, pr.num_levels, pr.pt_leaf, step_b(ip), ...
                opt.tolerance, string(opt.method), string(opt.rank_pruning), ...
                L, nnz(m), eqns, unknowns, rho, ...
                median(kap), kappa_mp, max(kap), median(nred_stat), n_vacuous, ...
                min(smin), median(res), max(res), median(iso), ...
                'VariableNames', {'problem', 'dim', 'n', 'num_levels', 'pt_leaf', ...
                'step_b', 'tolerance', 'method', 'rank_pruning', 'level', 'nodes', 'equations', ...
                'unknowns', 'rho', 'kappa_median', 'kappa_mp', 'kappa_max', ...
                'redundant_median', 'vacuous', 'sigma_min', 'resid_median', ...
                'resid_max', 'isotropy_median'})]; %#ok<AGROW>
        end
        node_rows = [node_rows; table( ...
            repmat(string(pr.label), size(rec, 1), 1), rec(:, 1), rec(:, 2), rec(:, 3), ...
            rec(:, 4), rec(:, 5), rec(:, 6), rec(:, 7), rec(:, 8), rec(:, 9), ...
            'VariableNames', {'problem', 'level', 'equations', 'unknowns', 'kappa', ...
            'sigma_min', 'resid', 'rhs_norm', 'n_redundant', 'isotropy'})]; %#ok<AGROW>
        if ~isempty(rnk)
            fprintf('  rank search: level / node dim / k before prune / k used / exit / max(LHS/RHS)\n');
            names = {'CAP-unconverged', 'cap-converged', 'converged'};
            for q = 1:size(rnk, 1)
                fprintf('    %5d %10d %14d %9d   %-16s %9.2f\n', ...
                    rnk(q,1), rnk(q,2), rnk(q,3), rnk(q,4), ...
                    names{rnk(q,5)}, rnk(q,6));
            end
            rank_rows = [rank_rows; table( ...
                repmat(string(pr.label), size(rnk, 1), 1), rnk(:, 1), rnk(:, 2), ...
                rnk(:, 3), rnk(:, 4), string(names(rnk(:, 5)))', rnk(:, 6), ...
                'VariableNames', {'problem', 'level', 'node_dim', 'k_before_prune', ...
                'k_used', 'exit', 'max_lhs_over_rhs'})]; %#ok<AGROW>
        end
        results(end+1) = struct('problem', pr.label, 'table', tbl, ...
            'summary', summary_rows(summary_rows.problem == string(pr.label), :)); %#ok<AGROW>
    end

    if ~isempty(opt.output_dir)
        if ~isfolder(opt.output_dir), mkdir(opt.output_dir); end
        writetable(summary_rows, fullfile(opt.output_dir, 'ls_conditioning_summary.csv'));
        writetable(node_rows, fullfile(opt.output_dir, 'ls_conditioning_nodes.csv'));
        writetable(rank_rows, fullfile(opt.output_dir, 'rank_search.csv'));
        save(fullfile(opt.output_dir, 'ls_conditioning.mat'), ...
            'summary_rows', 'node_rows', 'rank_rows', 'opt');
        fprintf('\nWrote ls_conditioning_summary.csv, ls_conditioning_nodes.csv and rank_search.csv to %s\n', ...
            opt.output_dir);
    end
end
