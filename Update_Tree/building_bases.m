function [U_b_i_all, Omega_to_use, Y_to_use, k_A, num_iterations, basis_stats] = building_bases(method, user_tolerance, b, mv, n, Y, Omega, ...
                num_blocks_side, block_or_pts_per_block, fixed_pt_leaf, node, current_level, num_levels, kernel_type, ...
                symmetric_kernel_list, dim, start_cols_Omega, collect_diagnostics, sample_distribution, perform_rank_pruning)
    % buildingBases: Find a rank k that meets the fixed user tolerance 
    % and builds U, Omega, and Y for the tree. This covers both leaf and 
    % non-leaf levels.

    if nargin < 16 || isempty(dim), dim = 1; end
    if nargin < 17, start_cols_Omega = []; end
    if nargin < 18, collect_diagnostics = false; end
    if nargin < 19 || isempty(sample_distribution), sample_distribution = 'real'; end
    if nargin < 20 || isempty(perform_rank_pruning), perform_rank_pruning = 'leaf'; end

    rank_pruning_scope = normalize_rank_pruning(perform_rank_pruning);
    perform_rank_pruning = strcmp(rank_pruning_scope, 'all') || ...
        (strcmp(rank_pruning_scope, 'leaf') && current_level == num_levels);

    method = lower(char(method));
    if ~ismember(method, ...
            {'naive', 'global', 'iterative', 'householder', 'safeguarded', ...
             'corrected'})
        error(['Unknown nullspace method "%s". Choose ''naive'', ' ...
            '''global'', ''iterative'', ''householder'', ' ...
            '''safeguarded'', or ''corrected''.'], method);
    end

    original_step_b = b;
    continue_iterations = true;

    basis_stats = struct( ...
        'method', method, ...
        'level', current_level, ...
        'rank', NaN, ...
        'rank_before_pruning', NaN, ...
        'pruned_count', NaN, ...
        'iterations', NaN, ...
        'num_nodes', NaN, ...
        'nullspace_time', 0, ...
        'diagnostics_time', 0, ...
        'max_nullspace_residual', NaN, ...
        'max_orthogonality_error', NaN, ...
        'min_initial_rcond', NaN, ...
        'safeguard_fallback_count', 0);
    max_nullspace_residual = 0;
    max_orthogonality_error = 0;
    min_initial_rcond = Inf;
    safeguard_fallback_count = 0;

    if dim == 1
        block_size = block_or_pts_per_block;
        num_blocks = num_blocks_side;
    elseif dim == 2
        pts_per_block = block_or_pts_per_block;
        block_size = start_cols_Omega;
        num_blocks = num_blocks_side^2;
    end

    % ===========================
    % Initialize the column pool
    % ===========================
    % We treat input Y and Omega as read-only static pools.
    % We only track the indices we are allowed to see (pool_view_limit).
    
    total_cols_in_pool = size(Omega, 2);

    if current_level == num_levels
        pool_view_limit = total_cols_in_pool;
    else
        % NON-LEAF LEVEL: Truncate view to block_size
        if size(Omega, 2) >= block_size
            pool_view_limit = block_size;
        else
            pool_view_limit = size(Omega, 2); 
        end
    end
    
    % Track where we are in the pool for fetching new columns
    pool_fetch_idx = pool_view_limit; 

    % Buffers for newly computed columns (that don't exist in the pool)
    Omega_new = zeros(size(Omega, 1), 0, 'like', Omega);
    Y_new = zeros(size(Y, 1), 0, 'like', Y);

    % The stopping test holds out the most recent block of b columns, so the
    % accepted basis can only grow to block_size - b. A step size close to the
    % node dimension therefore leaves the search one or two all-or-nothing
    % decisions: it either accepts far more rank than the node needs, or
    % reaches the node dimension with the test still failing and returns an
    % uncompressed level. Clamping b to a quarter of the node dimension keeps
    % at least four resolvable steps. Leaf nodes are large enough that this
    % never binds; it acts only on the small coarse nodes that appear once the
    % finer levels have compressed well.
    b_cap = max(1, floor(block_size / 4));
    if b > b_cap
        original_b_val = b;
        b = b_cap;
        original_step_b = b;

        fprintf("\n");
        warning("Step size b = %d exceeds a quarter of the block size = %d.\n" + ...
            "Step size 'b' has been reset to max(1, floor(block_size / 4)) = %d so the rank search has resolvable steps.\n", ...
            original_b_val, block_size, b);
    end

    % --- First Iteration ---
    iter = 1;

    add_next_batch_cols(b);

    Q_i_all = cell(num_blocks,1); 
    U_b_i_all = cell(num_blocks,1);
    % Width of each node's accepted basis, maintained through the loop so
    % that the level's common rank can be the widest node rather than the
    % iteration count times b (see the k_A assignment after the loop).
    accepted_width = zeros(num_blocks, 1);
    Y_Q_norm_all = zeros(num_blocks, 1);
    A_12_Psi_all = cell(num_blocks, 1);
    LU_struct_all = cell(num_blocks, 1);
    initial_rcond_all = nan(num_blocks, 1);
    LHS_all = zeros(num_blocks, 1);
    % Which nodes failed the stopping test on the most recent trial. Needed
    % to tell a node that has exhausted its rank budget from one that is
    % merely marginal; see the exit branch after the loop.
    still_failing_all = false(num_blocks, 1);
    RHS_all = zeros(num_blocks, 1);
    % Same threshold without the level-dependent relaxation. The relaxation
    % exists to cap rank growth at coarse levels, where the inherited error
    % already exceeds eps; feeding it to the pruning budget would additionally
    % license more aggressive pruning at exactly those levels, compounding the
    % error the relaxation is acknowledging. Pruning uses this instead.
    RHS_strict_all = zeros(num_blocks, 1);

    if dim == 2
        % ========================= 2D =========================
        %-------------------------------
        % IF NOT AT LEAF, BUILD A MAP FOR EACH CHILD'S SKELETEN INDICES
        %-------------------------------
        if current_level < num_levels
            %-------------------------------
            % INDEX MAPPING
            %-------------------------------
            nextL = current_level + 1;
            start_nextL = 1 + sum(4.^(0:(nextL-2)));
            end_nextL   =   sum(4.^(0:(nextL-1)));
    
            num_parents_side  = 2^(current_level - 1);
            num_children_side = 2^(nextL - 1);
    
            children_ids = zeros((end_nextL - start_nextL + 1), 1);
            idx = 1;
            for parent_block_id = 0:(num_parents_side^2 - 1)
                prow = mod(parent_block_id, num_parents_side);
                pcol = floor(parent_block_id / num_parents_side);
    
                child_rows = [2*prow,     2*prow + 1,   2*prow,     2*prow + 1];
                child_cols = [2*pcol,     2*pcol,       2*pcol + 1, 2*pcol + 1];
                
                for ii = 1:4
                    crow = child_rows(ii);
                    ccol = child_cols(ii);
                    children_ids(idx) = start_nextL + crow + ccol * num_children_side;
                    idx = idx + 1;
                end
            end
    
            total_skel = 0; % to preallocate all_skel
            for ii = 1:numel(children_ids)
                cid = children_ids(ii);
                total_skel = total_skel + numel(node(cid).Indices.skeleton);
            end
    
            all_skel = zeros(total_skel, 1); 
    
            offset = 0; 
            for ii = 1:numel(children_ids)
                cid = children_ids(ii);
                skel_list = node(cid).Indices.skeleton(:); 
                nrows = numel(skel_list);
                all_skel(offset+1 : offset+nrows) = skel_list;
                offset = offset + nrows;
            end
        end
    end

    % ===========================
    % Pre-compute all 2D block indices
    % ===========================
    indices_all = cell(num_blocks, 1);
    for block = 1:num_blocks
        if dim == 1
            index_1 = (block-1)*block_size + 1;
            index_2 = block*block_size;
            indices_all{block} = index_1:index_2;

        elseif dim == 2
            [row_idx, col_idx] = ind2sub([num_blocks_side, num_blocks_side], block);
            if current_level == num_levels
                row_start = (row_idx - 1) * pts_per_block + 1;
                row_end   =  row_idx   * pts_per_block;
                col_start = (col_idx - 1) * pts_per_block + 1;
                col_end   =  col_idx   * pts_per_block;
    
                [cols_grid, rows_grid] = meshgrid(col_start:col_end, row_start:row_end);
                indices_all{block} = sub2ind([sqrt(n), sqrt(n)], rows_grid(:), cols_grid(:))';
            else
                num_children_side = 2^(nextL-1);
                prow = row_idx - 1;
                pcol = col_idx - 1;
    
                child_rows = [2*prow, 2*prow + 1,   2*prow, 2*prow+1];
                child_cols = [2*pcol, 2*pcol, 2*pcol+1,   2*pcol+1];
    
                child_ids = zeros(1,4);
                for ii = 1:4
                    crow = child_rows(ii);
                    ccol = child_cols(ii);
                    child_ids(ii) = start_nextL + crow + ccol * num_children_side;
                end
    
                block_globals = [];
                for cid = child_ids
                    block_globals = [block_globals; node(cid).Indices.skeleton(:)];
                end
    
                [~, locs] = ismember(block_globals, all_skel);
                indices_all{block} = locs';  
            end
        end
    end
    
    for block = 1:num_blocks
        % --- Use 2D pre-computed indices ---
        indices = indices_all{block};

        % Construct local block from (Pool view) + (New buffer)
        Y_i = [Y(indices, 1:pool_view_limit), Y_new(indices, :)];
        Omega_i = [Omega(indices, 1:pool_view_limit), Omega_new(indices, :)];

        nullspace_timer = tic;
        Omega_i_fixed = Omega_i(:, 1:block_size);
        Omega_i_new = Omega_i(:, block_size+1:end);
        correction_options = struct('target_tolerance', user_tolerance);
        [Q_i, nullspace_state] = initialize_nullspace_basis( ...
            method, Omega_i_fixed, Omega_i_new, correction_options);
        LU_struct_all{block} = nullspace_state;
        safeguard_fallback_count = safeguard_fallback_count + ...
            nullspace_state.full_recompute_count;
        initial_rcond_all(block) = nullspace_state.initial_rcond;
        basis_stats.nullspace_time = basis_stats.nullspace_time + toc(nullspace_timer);
        record_nullspace_diagnostics(Omega_i, Q_i, initial_rcond_all(block));

        Q_i_all{block} = Q_i;
    
        Y_Q_i = Y_i * Q_i;
        Y_Q_norm_all(block) = Y_Q_norm_all(block) + norm(Y_Q_i, 'fro')^2;
    
        [U_b, ~] = qr(Y_Q_i, 'econ'); 
        U_b = U_b(:, 1:b);
        U_b_i_all{block} = U_b; 
    end
    
    % --- Second Iteration Onward ---
    if continue_iterations
        iter = 2;
    end

    b_prev = b;
    b_updated = b;

    reach_max_block_size = false;
    
    while b_updated + b <= block_size || continue_iterations
        b_updated = b_updated + b;
    
        continue_iterations = false;

        % Update Y and Omega
        % --- Figure out how many columns to add ---
        if b_updated > block_size
            num_cols_to_add = block_size - b_prev;
            if num_cols_to_add <= 0 % Safety check if we already hit block size
                num_cols_to_add = 0;
            else
                b = num_cols_to_add;
            end
            reach_max_block_size = true;
        else
            num_cols_to_add = b;
        end
        
        % --- Get num_cols_to_add columns ---
        if num_cols_to_add > 0
            add_next_batch_cols(num_cols_to_add);
        end
    
        for block = 1:num_blocks
            % --- Use pre-computed 2D indices ---
            indices = indices_all{block};
            
            Y_i = [Y(indices, 1:pool_view_limit), Y_new(indices, :)];
            Omega_i = [Omega(indices, 1:pool_view_limit), Omega_new(indices, :)];

            nullspace_timer = tic;
            Omega_i_new = Omega_i(:, end-b+1:end);
            previous_recompute_count = ...
                LU_struct_all{block}.full_recompute_count;
            [Q_i, Q_i_last_b, nullspace_state] = extend_nullspace_basis( ...
                method, Omega_i, Q_i_all{block}, LU_struct_all{block}, Omega_i_new);
            LU_struct_all{block} = nullspace_state;
            Q_i_all{block} = Q_i;
            safeguard_fallback_count = safeguard_fallback_count + ...
                nullspace_state.full_recompute_count - previous_recompute_count;
            basis_stats.nullspace_time = basis_stats.nullspace_time + toc(nullspace_timer);

            if reach_max_block_size
                b_updated = block_size;
            end

            recomputed_full = strcmp(method, 'naive') || ...
                nullspace_state.recomputed_full;
            if recomputed_full
                % Restart the local linear algebra from all retained samples.
                % The first columns define the candidate basis; the final
                % block is held out for the stopping test.
                G_all = Y_i * Q_i;
                accepted_rank = size(Q_i, 2) - b;
                accepted_width(block) = accepted_rank;
                [U_b_i, ~] = qr(G_all(:, 1:accepted_rank), 'econ');
                U_b_i = U_b_i(:, 1:accepted_rank);
                A_12_Psi_b = G_all(:, accepted_rank+1:end);
                Y_Q_norm_all(block) = norm(G_all, 'fro')^2;
            else
                % Compute only the newly added sample block.
                A_12_Psi_b = Y_i * Q_i_last_b;
                U_b_i = U_b_i_all{block};
                Y_Q_norm_all(block) = Y_Q_norm_all(block) + ...
                    norm(A_12_Psi_b, 'fro')^2;
            end
            record_nullspace_diagnostics(Omega_i, Q_i, initial_rcond_all(block));
            
            A_12_Psi_all{block} = A_12_Psi_b;
    
            % --- Check tolerance errors --- 
            % residual = (eye(block_size) - U_b_i * U_b_i') * A_12_Psi_b;
            residual = A_12_Psi_b - U_b_i * (U_b_i' * A_12_Psi_b); clear A_12_Psi_b;
            LHS_estimated = sqrt(1/b) * norm(residual, 'fro');
            LHS_all(block) = LHS_estimated;

            % TOL_RELAX_OVERRIDE (global) forces both factors to one value,
            % so the level-dependent relaxation can be switched off for
            % experiments without editing the defaults.
            global TOL_RELAX_OVERRIDE %#ok<GVMIS>
            if ~isempty(TOL_RELAX_OVERRIDE) && isnumeric(TOL_RELAX_OVERRIDE)
                tol_relax_gentle = TOL_RELAX_OVERRIDE;
                tol_relax_aggressive = TOL_RELAX_OVERRIDE;
            elseif dim == 1
                tol_relax_gentle = sqrt(2);
                tol_relax_aggressive = sqrt(2);
            else
                tol_relax_gentle = 2;
                tol_relax_aggressive = 2;
            end
            
            split_level = ceil(num_levels / 2);

            if current_level <= split_level
                tol_relax_factor = tol_relax_aggressive;
            else
                tol_relax_factor = tol_relax_gentle;
            end

            tolerance_mod = tol_relax_factor^(num_levels - current_level);

            % Freeze rule (default since 2026-09-26, Sec. 4.4 of the paper):
            % besides its own relative test, a node is frozen once its
            % residual is below the level's absolute tolerance
            % eps/sqrt(b) * sqrt(mean sampled energy over the level), i.e.
            % the test reads max(eta_i, eta_bar_l). A block whose norm sits
            % at matvec roundoff -- a leaf where the compactly supported
            % cavity potential of the Lippmann-Schwinger problem vanishes
            % -- can never pass the purely relative test: its threshold is
            % eps times roundoff, while its residual is roundoff that
            % shrinks only as each batch absorbs b of the node's noise
            % dimensions, so it runs to full rank and, the level width
            % being common, drags the whole level dense (30 to 64 of 256
            % leaves at N=256^2). Under the rule such a node passes its
            % first test and is stored without compression. The rule at
            % most doubles sum_i eta_i^2, so the level's total error bound
            % grows by at most sqrt(2). Implemented as a floor on the
            % node's energy in RHS below. The global STOP_FLOOR_MODE =
            % 'none' switches it off for diagnostics
            % (run_stop_floor_diagnostic.m); 'level_mean_over_n' is the
            % narrower reference, evaluated and rejected because it loses
            % its margin at coarse levels
            % (diagnostics/run_stop_floor_diagnostic.m).
            global STOP_FLOOR_MODE %#ok<GVMIS>
            node_energy = Y_Q_norm_all(block);
            if isempty(STOP_FLOOR_MODE) || strcmp(STOP_FLOOR_MODE, 'level_mean')
                node_energy = max(node_energy, mean(Y_Q_norm_all));
            elseif strcmp(STOP_FLOOR_MODE, 'level_mean_over_n')
                % Narrower variant under evaluation: the floor is the mean
                % divided by the number of nodes, so only nodes far below
                % the level are relaxed; sum_i eta_i^2 grows by at most
                % the factor 1 + 1/num_blocks instead of 2.
                node_energy = max(node_energy, mean(Y_Q_norm_all) / numel(Y_Q_norm_all));
            end
            RHS = user_tolerance * tolerance_mod * sqrt(1/(original_step_b*iter)) * sqrt(node_energy);
            RHS_all(block) = RHS;
            RHS_strict_all(block) = RHS / max(tolerance_mod, realmin);

            % --- Optional rank-search trace (diagnostic; inert by default) ---
            % Set the global RANK_TRACE to a struct with .enabled true to
            % record one row per node per trial: the basis width, the two
            % sides of the stopping test, how well the nullspace basis
            % actually annihilates the local samples, and the orthogonality
            % of both bases. Used to diagnose nodes that never converge.
            global RANK_TRACE %#ok<GVMIS>
            if ~isempty(RANK_TRACE) && isstruct(RANK_TRACE) && RANK_TRACE.enabled ...
                    && (isempty(RANK_TRACE.level) || current_level == RANK_TRACE.level)
                tr_null = norm(Omega_i * Q_i, 'fro') / max(norm(Omega_i, 'fro'), realmin);
                tr_q = norm(Q_i' * Q_i - eye(size(Q_i, 2)), 'fro');
                tr_u = norm(U_b_i' * U_b_i - eye(size(U_b_i, 2)), 'fro');
                RANK_TRACE.rows(end+1, :) = [current_level, block, iter, ...
                    size(U_b_i, 2), LHS_estimated, RHS, tr_null, tr_q, tr_u];
            end

            
            % --- Two thresholds ---
            % The level stops when no node exceeds RHS. A node stops GROWING
            % only when its estimate is comfortably below RHS, not merely
            % below it. The estimate is a b-column Gaussian estimate of the
            % residual norm with a relative standard error near 1/sqrt(2b),
            % 18%% at b = 16, so a node parked at a rank where its true error
            % sits at the threshold passes and fails at random.
            %
            % Before 2026-09-12 no node was ever frozen: every node grew by b
            % on every trial, so nothing lingered at a marginal rank and the
            % level finished in two or three trials. Freezing was introduced
            % to stop a converged node being fed noise, which destroys its
            % basis at the roundoff floor. Freezing on a bare pass kept both
            % problems' worst features: marginal nodes stopped growing, the
            % level could not finish until all of them passed in the same
            % trial, and the operator count grew with the node count (64
            % leaves 304 products, 1024 leaves 512).
            %
            % A margin separates the two cases cleanly. A genuinely converged
            % node sits four or five orders below the threshold and freezes
            % at once; a marginal node sits within a factor of two of it and
            % grows, as it did before freezing existed. FREEZE_MARGIN
            % overrides the factor for experiments. The default of 0.25 is
            % several standard errors of the estimate at either block size,
            % and it returns the 1D operator count to a flat 304 at every N.
            global FREEZE_MARGIN %#ok<GVMIS>
            if ~isempty(FREEZE_MARGIN) && isnumeric(FREEZE_MARGIN)
                freeze_margin = FREEZE_MARGIN;
            else
                freeze_margin = 0.25;
            end
            grow_threshold = freeze_margin * RHS;

            % The level stops when no node exceeds RHS; a node stops
            % GROWING only once it is comfortably below RHS. The two
            % thresholds are separate on purpose, and a single threshold at
            % either value will not do: at RHS a node parked at a marginal
            % rank passes and fails at random and the level cannot finish,
            % and at grow_threshold the two collapse into one tighter
            % tolerance with the same effect. Measured on 1D Poisson at
            % eps = 1e-9, a single threshold at grow_threshold took the
            % operator count to 304, 352, 448 for 64, 256 and 1024 leaves
            % against a flat 304.
            %
            % Deferring the append to the end of the trial -- forming the
            % candidate extensions but committing them only if the level
            % continues -- was tried on 2026-09-16 and reverted. It makes the
            % look-ahead genuinely independent, but a node in
            % (grow_threshold, RHS] is passing and NOT comfortably converged,
            % so discarding its candidate leaves it sitting at the tolerance
            % rather than below it, and that accumulates over levels. At
            % eps = 1e-3, where a level converges in one or two trials and a
            % discarded block is most of the basis, the Helmholtz strip at
            % N = 4096 went from a relative residual of 0.10 to 3.5, that is
            % worse than returning zero, with the rank falling from 9 to 5.
            still_failing_all(block) = LHS_estimated > RHS;
            if LHS_estimated > RHS
                continue_iterations = true;
                status = "CONTINUE";
            else
                status = "CONVERGED";
            end

            % --- Logging ---
            if block <= 3 
                fprintf('Lvl %2d | Blk %3d | Iter %d | LHS: %.2e | RHS: %.2e | Status: %s\n', ...
                        current_level, block, iter, LHS_estimated, RHS, status);
            end
    
            if recomputed_full
                % Rebuild the complete trial basis, including the held-out
                % block, for use if another rank trial is required.
                [U_b_i, ~] = qr(G_all, 'econ');
                U_b_i_all{block} = U_b_i(:, 1:size(Q_i, 2));
                clear G_all U_b_i residual;
            elseif LHS_estimated > grow_threshold
                % Not comfortably converged: append the directions the new
                % sample revealed, orthogonalized against the retained basis.
                [U_b_extend, ~] = qr(residual, 'econ'); clear residual;
                U_b_extend = U_b_extend(:, 1:b);
                [U_b_extend, ~] = qr(U_b_extend - U_b_i * (U_b_i' * U_b_extend), 'econ'); %reorthogonalize
                U_b_i = [U_b_i, U_b_extend]; clear U_b_extend;
                U_b_i_all{block} = U_b_i; clear U_b_i;
                accepted_width(block) = size(U_b_i_all{block}, 2);
            else
                % Comfortably converged: leave this node's basis alone, as
                % Algorithm 4.3 does (U_new = U_old). It is tested again with
                % the next sample and resumes growing only if that test fails.
                %
                % Appending to a node that has already converged is what used
                % to happen here, and it is destructive: once the node's
                % residual sits at roundoff, the appended columns are noise,
                % a single reorthogonalization cannot make them orthogonal to
                % U, and ||U'U - I|| climbs from 1e-14 to O(1) within a few
                % batches, after which the stopping estimate is meaningless
                % and the level runs to full rank. Measured on the Matern
                % (1.5, 0.1) kernel at eps = 1e-12, leaf 1024, b = 8: node 1
                % passed at batch 39, and its estimate went 4e-11 -> 1e-4 ->
                % 1e+2 over batches 53-56 while naive recomputation certified
                % at rank 512. Nodes frozen early are padded to the level's
                % common width after the loop.
                accepted_width(block) = size(U_b_i_all{block}, 2);
                clear residual U_b_i;
            end
    
        end
    
        if b_updated >= block_size
            break;
        elseif ~ continue_iterations
            break;
        end
        
        b_prev = b_updated;
        iter = iter + 1;
    end

    % A node has genuinely exhausted its rank budget only if it is still
    % failing AND its accepted basis has reached the widest certifiable
    % width. Any other node that happens to fail on the final trial is
    % merely marginal: it grew by b in the branch above and its basis is
    % nowhere near the ceiling.
    rank_ceiling = max(block_size - original_step_b, 0);
    exhausted_nodes = still_failing_all & (accepted_width >= rank_ceiling);

    if b_updated >= block_size && continue_iterations && ...
            ~level_exhausted_rank_budget(still_failing_all, accepted_width, ...
                block_size, original_step_b)
        % The sample budget ran out while at least one node was still
        % marginal, but no node came close to the ceiling. This used to be
        % treated as failure at max rank, which stamped the WHOLE level at
        % the node dimension and discarded every accepted basis.
        %
        % Measured on 1D Poisson at N = 262144, eps = 1e-9, b = 16, seed 0:
        % of 1024 leaves exactly one (node 501) was marginal, oscillating at
        % LHS/RHS between 0.41 and 0.85 with a basis of 16 columns, and on
        % the sixteenth and final trial its random estimate came in at 1.03.
        % The level was then stored at rank 256 rather than 32: 1189 MB
        % instead of 613, and 672 operator applications instead of 304.
        % Exact recomputation of the nullspace, which never freezes a node,
        % converged the same configuration at rank 32.
        num_iterations = iter;
        k_A = max(accepted_width);
        b = original_step_b;
        marginal = sum(still_failing_all);
        fprintf("\n");
        warning("Rank search used its whole sample budget at level %d with " + ...
            "%d of %d nodes still marginal, none of them near the rank " + ...
            "ceiling (%d). Keeping the widest accepted basis, %d columns. " + ...
            "Those nodes are resolved at approximately, rather than below, " + ...
            "the requested tolerance.", current_level, marginal, num_blocks, ...
            rank_ceiling, k_A);
    elseif b_updated >= block_size && continue_iterations
        % Failure at max rank: the search reached the node dimension with the
        % stopping test still failing, so the level is returned at full rank
        % without ever satisfying the requested tolerance.
        %
        % WHY THIS HAPPENS. The stopping test holds out the most recent block
        % of b columns (accepted_rank = size(Q_i,2) - b), so the largest basis
        % the search can ever certify on a node of dimension m is m - b. A
        % block whose numerical rank exceeds m - b therefore always ends here.
        %
        % THIS IS USUALLY THE RIGHT ANSWER, NOT A FAILURE. Measured on 2D
        % Poisson at 128^2 and eps = 1e-12, whose leaf blocks have numerical
        % rank 245 of 256: at b = 32 and b = 16 the level is stored dense and
        % the residual is 2.1e-11 and 2.7e-11; at b = 8 the search can certify
        % rank 245 and does, and the result costs 6.6%% MORE storage, because
        % a rank-245 factorization stores a basis a dense block does not need,
        % and the residual is 2.7 times worse. Matern (1.5, 0.1) at 128^2
        % behaves the same way: dense at 455.2 MB, compressed to rank 230 at
        % 496.9 MB. So a dense level at a tight tolerance means the block is
        % not low rank there, and storing it dense is both correct and cheap.
        %
        % WHAT TO DO. This is almost always a leaf size chosen too small for
        % the kernel. The leaf must exceed its own off-diagonal numerical rank
        % by more than b. Measured on the 2D Matern kernel (nu = 0.5,
        % ell = 0.5) at eps = 1e-6: leaf 64 fails here with LHS/RHS = 5.7,
        % leaf 144 converges at rank 90, leaf 256 converges at rank 116. At
        % eps = 1e-9 the leaf-64 failure is far worse (LHS/RHS = 6682) because
        % the rank needed grows with the tolerance. The Helmholtz kernel
        % behaves the same way at leaf 64 and eps = 1e-9.
        num_iterations = iter;
        k_A = block_size;
        fprintf("\n");
        warning("Level %d is not low rank at this tolerance: %d of %d nodes " + ...
            "still fail the stopping test with a basis at the widest " + ...
            "certifiable width (%d of %d), so the level is stored dense. " + ...
            "This is usually the correct outcome rather than a failure; a " + ...
            "dense block is both exact and, at these ranks, cheaper to " + ...
            "store than its own factorization.", ...
            current_level, sum(exhausted_nodes), num_blocks, ...
            rank_ceiling, block_size);
    elseif b_updated >= block_size && ~ continue_iterations
        % Success at max rank
        num_iterations = iter - 1;
        k_A = max(accepted_width);
        b = original_step_b;
    else
        % Success at early rank. The common rank of the level is the widest
        % accepted basis any node reached, not the number of trials times b.
        % A node that passed its test is frozen but re-tested with every
        % later sample, and it may fail one of those re-tests and grow by b;
        % while marginal nodes do this in turn the level keeps iterating, and
        % tying k_A to the iteration count inflated every node to that count
        % times b. Measured on 1D Poisson at b = 16, eps = 1e-9, N = 16384:
        % the leaf level ran five trials and was stored at rank 64 (80 before
        % pruning) although no node had accepted more than 32 columns and the
        % numerical rank is about 19. J_l * b remains an upper bound on k_A.
        num_iterations = iter-1;
        k_A = max(accepted_width);
    end

    % Nodes that passed the stopping test before the level finished were
    % frozen at their own width. Pad them to the common width k_A. The
    % samples such a node received after it stopped growing carry exactly
    % the directions its basis still misses, down to roundoff, so they are
    % used for the padding: one QR of [U, G] per frozen node, in place of
    % the arbitrary complement directions the full QR at the end of
    % Algorithm 4.3 would supply. Padding with arbitrary directions leaves
    % eps-level rows in the parent's off-diagonal block that the parent
    % then has to resolve; measured on Matern (1.5, 0.1) at eps = 1e-12,
    % leaf 1024, that cost 72 extra ranks at the next level and 3.5% storage
    % against naive recomputation, which informed padding recovers. The
    % final held-out block is excluded so the pruning look-ahead below stays
    % independent of the basis it certifies.
    for block = 1:num_blocks
        w = size(U_b_i_all{block}, 2);
        if w < k_A
            indices = indices_all{block};
            Y_i = [Y(indices, 1:pool_view_limit), Y_new(indices, :)];
            G_all = Y_i * Q_i_all{block};
            held_out = size(A_12_Psi_all{block}, 2);
            % The basis already spans the first w sample directions; only
            % the later ones carry new information. Passing the early ones
            % too would make the QR emit noise directions for them first.
            G_late = G_all(:, w+1:end-held_out);
            U_b_i_all{block} = pad_basis_to_width(U_b_i_all{block}, k_A, G_late);
            clear Y_i G_all G_late;
        end
    end

    %-------------------------------
    % ENERGY-ORDERED RANK PRUNING
    %-------------------------------

    k_before_pruning = k_A;
    fprintf("The k before rank pruning is %d.\n", k_before_pruning);


    if perform_rank_pruning
        % Pruning at one level raises the error entering the next, and the
        % per-level stopping test does not account for that. Because every
        % transformation is unitary the error accumulates additively over the
        % L levels, so each level should spend at most eps/L of the norm
        % budget. The budget here is on squared energies, hence 1/L^2.
        budget_fraction = 1 / max(num_levels, 1)^2;
        [U_b_i_all, k_A] = refine_rank_sub_block( ...
            U_b_i_all, LHS_all, RHS_all, A_12_Psi_all, ...
            b, k_A, num_blocks, false, budget_fraction, RHS_strict_all);
    else
        for block = 1:num_blocks
            U_b_i_all{block} = U_b_i_all{block}(:, 1:k_A);
        end
    end
    
    fprintf("The k that we use is %d.\n", k_A);

    % The complete nullspace bases and update states are not needed after
    % rank selection. Release them before concatenating the potentially
    % large output sample matrices to reduce peak memory.
    record_rank_diagnostic(current_level, block_size, k_before_pruning, k_A, ...
        b_updated, continue_iterations, LHS_all, RHS_all);

    clear Q_i_all LU_struct_all A_12_Psi_all LHS_all RHS_all RHS_strict_all ...
        Y_Q_norm_all indices_all Y_i Omega_i Omega_i_fixed Omega_i_new ...
        Q_i Q_i_last_b nullspace_state residual U_b_i G_all;

    % --- Construct output matrices (one-time copy) ---
    % Combine the static pool view, the new buffer, and any unused padding.
    current_cols = pool_view_limit + size(Omega_new, 2);

    if current_level ~= num_levels && current_cols < total_cols_in_pool
        fprintf('Padding output matrices. Current cols: %d. Pool cols: %d.\n', ...
            current_cols, total_cols_in_pool);
        Omega_to_use = [Omega(:, 1:pool_view_limit), Omega_new, ...
            Omega(:, pool_view_limit+1:end)];
        Y_to_use = [Y(:, 1:pool_view_limit), Y_new, ...
            Y(:, pool_view_limit+1:end)];
    else
        Omega_to_use = [Omega(:, 1:pool_view_limit), Omega_new];
        Y_to_use = [Y(:, 1:pool_view_limit), Y_new];
    end

    basis_stats.rank = k_A;
    basis_stats.rank_before_pruning = k_before_pruning;
    basis_stats.pruned_count = k_before_pruning - k_A;
    basis_stats.iterations = num_iterations;
    basis_stats.num_nodes = num_blocks;
    basis_stats.safeguard_fallback_count = safeguard_fallback_count;
    if collect_diagnostics
        basis_stats.max_nullspace_residual = max_nullspace_residual;
        basis_stats.max_orthogonality_error = max_orthogonality_error;
        if isfinite(min_initial_rcond)
            basis_stats.min_initial_rcond = min_initial_rcond;
        end
    end

    % --- NESTED HELPER FUNCTION ---
    function add_next_batch_cols(num_cols_to_add)
        % This function updates the 'pool_view_limit' or computes new columns
        % and appends them to 'Omega_new' / 'Y_new'.
        % It avoids creating copies of the pool.
        
        if pool_fetch_idx + num_cols_to_add <= total_cols_in_pool
            % YES. Get them from the pool (Zero copy - Just update index).
            fprintf('Cache: Using pool columns %d to %d.\n', pool_fetch_idx + 1, pool_fetch_idx + num_cols_to_add);
            
            % We just extend the view of the pool we are allowed to use
            pool_view_limit = pool_view_limit + num_cols_to_add;
            pool_fetch_idx = pool_fetch_idx + num_cols_to_add; 
            
        else
            % NO. Get what we can, compute the rest.
            cols_from_pool = max(0, total_cols_in_pool - pool_fetch_idx);
            
            if cols_from_pool > 0
                pool_view_limit = pool_view_limit + cols_from_pool;
                pool_fetch_idx  = pool_fetch_idx + cols_from_pool;
            end
            
            cols_to_compute = num_cols_to_add - cols_from_pool;
            fprintf('Cache: Pool empty. Computing %d new columns.\n', cols_to_compute);

            if cols_to_compute > 0
                if current_level == num_levels
                    [Om_n, Y_n] = add_columns_at_leaves(cols_to_compute, mv, n, sample_distribution);
                else
                    [Y_n, Om_n] = expand_Omega_Y(node, current_level, cols_to_compute, mv, n, fixed_pt_leaf, kernel_type, symmetric_kernel_list, dim, sample_distribution);
                end
                
                % Append to the new buffer (save memory)
                Omega_new = [Omega_new, Om_n];
                Y_new     = [Y_new, Y_n];
            end
            
            % Pool index is now effectively at end or beyond
            pool_fetch_idx = total_cols_in_pool; 
        end
    end

    function record_nullspace_diagnostics(Omega_local, Q_local, rcond_estimate)
        if ~collect_diagnostics
            return;
        end

        diagnostics_timer = tic;
        omega_norm = max(norm(Omega_local, 'fro'), realmin);
        nullspace_residual = norm(Omega_local * Q_local, 'fro') / omega_norm;
        orthogonality_error = norm( ...
            Q_local' * Q_local - eye(size(Q_local, 2), 'like', Q_local), 'fro');

        max_nullspace_residual = max(max_nullspace_residual, nullspace_residual);
        max_orthogonality_error = max(max_orthogonality_error, orthogonality_error);
        if isfinite(rcond_estimate)
            min_initial_rcond = min(min_initial_rcond, rcond_estimate);
        end
        basis_stats.diagnostics_time = basis_stats.diagnostics_time + toc(diagnostics_timer);
    end
    % --- END OF HELPER FUNCTION ---
end

function U = pad_basis_to_width(U, width, G)
%PAD_BASIS_TO_WIDTH Append orthonormal columns to U up to WIDTH.
%   The retained columns are kept as they are. If sample directions G are
%   given, the appended columns are the part of G outside span(U), taken
%   from a Householder QR of [U, G] so they are orthonormal to working
%   precision however small that part is. Any remaining width comes from
%   the complement of U, the trailing columns of its full orthogonal factor.
    [m, w] = size(U);
    width = min(width, m);
    if w >= width
        return;
    end
    if nargin >= 3 && ~isempty(G)
        [Q_cat, ~] = qr([U, G], 'econ');
        take = min(width, size(Q_cat, 2));
        U = [U, Q_cat(:, w+1:take)];
        w = size(U, 2);
        if w >= width
            return;
        end
    end
    [Q_full, ~] = qr(U);
    U = [U, Q_full(:, w+1:width)];
end

function scope = normalize_rank_pruning(value)
    if islogical(value) && isscalar(value)
        if value
            scope = 'all';
        else
            scope = 'none';
        end
        return;
    end
    scope = lower(char(value));
    if ~ismember(scope, {'all', 'leaf', 'none'})
        error(['Unknown rank-pruning scope "%s". Choose ''all'', ' ...
            '''leaf'', or ''none''.'], scope);
    end
end

function record_rank_diagnostic(level, block_size, k_pre, k_A, b_updated, ...
        continue_iterations, LHS_all, RHS_all)
%RECORD_RANK_DIAGNOSTIC Log how the adaptive rank search terminated at a level.
%   exit_reason 1: the rank reached the node dimension while the stopping test
%               was still failing -- the "failure at max rank" branch. Nothing
%               is compressed at this level and every index becomes skeleton.
%   exit_reason 2: the rank reached the node dimension with the test satisfied.
%   exit_reason 3: the test was satisfied below the node dimension.
%   The ratio is max_i LHS_i / RHS_i at exit: how far the worst node was from
%   the stopping threshold. Inert unless RANK_DIAG.enabled is set.
    global RANK_DIAG %#ok<GVMIS>
    if isempty(RANK_DIAG) || ~isstruct(RANK_DIAG) || ~RANK_DIAG.enabled
        return;
    end
    if b_updated >= block_size
        if continue_iterations
            reason = 1;
        else
            reason = 2;
        end
    else
        reason = 3;
    end
    ratio = NaN;
    if ~isempty(LHS_all) && ~isempty(RHS_all)
        r = LHS_all(:) ./ max(RHS_all(:), realmin);
        r = r(isfinite(r));
        if ~isempty(r), ratio = max(r); end
    end
    RANK_DIAG.rows(end+1, :) = [level, block_size, k_pre, k_A, reason, ratio];
end
