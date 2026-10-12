function node = store_node_results(node, Omega_final, Y_final, U_full, block_or_pts_per_block, level, num_levels, ...
    k, Omega_final_n, Y_final_n, V_full, kernel_type, symmetric_kernel_list, spd_kernel_list, dim, n1d)
    % storeNodeResults: Updates node(nd) with U, V, Y_tilde, Omega_tilde,
    %                   Y_tilde_n, Omega_tilde_n

    if nargin < 15 || isempty(dim), dim = 1; end

    if dim == 1
        % ========================= 1D =========================
        block_size = block_or_pts_per_block;

        % Nodes indices
        start_idx = 2^(level - 1);
        end_idx = 2^level - 1;

    elseif dim == 2
        % ========================= 2D =========================
        pts_per_block = block_or_pts_per_block;

        % Nodes indices
        start_idx = sum(4.^(0:level-2)) + 1;
        end_idx = sum(4.^(0:level-1));
    
        num_blocks_side = 2^(level - 1);

        if level < num_levels
            %-------------------------------
            % INDEX MAPPING
            %-------------------------------
            nextL = level + 1;
            start_nextL = 1 + sum(4.^(0:(nextL-2)));
            end_nextL   =   sum(4.^(0:(nextL-1)));
    
            num_parents_side  = 2^(level - 1);
            num_children_side = 2^(nextL - 1);
    
            children_ids = zeros((end_nextL - start_nextL + 1), 1);
            idx = 1;
            for parent_block_id = 0:(num_parents_side^2 - 1)
                prow = mod(parent_block_id, num_parents_side);
                pcol = floor(parent_block_id / num_parents_side);
    
                % Compute four children in column‐major order:
                %   (child_col=0,child_row=0), (child_col=0,child_row=1),
                %   (child_col=1,child_row=0), (child_col=1,child_row=1)
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
    
            all_skel = zeros(total_skel, 1); % will collect all skeleton indices from all children in order
    
            offset = 0;
            % Loop over every child node at level+1 to build the “all_skel” list
            for ii = 1:numel(children_ids)
                cid = children_ids(ii);
                skel_list = node(cid).Indices.skeleton(:);  % global row indices for that child's skeleton
                nrows = numel(skel_list);
    
                all_skel(offset+1 : offset+nrows) = skel_list;
                offset = offset + nrows;
            end

        end
    end

    % --- Start building node struct for the current level ---
    for nd = start_idx : end_idx
        if dim == 1
            % ========================= 1D =========================
            blk_idx = nd - start_idx;
            idx_start = blk_idx * block_size + 1;
            idx_end = (blk_idx + 1) * block_size;
    
            if level == num_levels
                node(nd).Indices.original = idx_start:idx_end;
            else
                left_child = 2*nd;
                right_child = 2*nd+1;
               
                node(nd).Indices.original = union(node(left_child).Indices.skeleton, node(right_child).Indices.skeleton);
            end
    
            if ~ismember(kernel_type, symmetric_kernel_list)
                Omega_i_n = Omega_final_n(idx_start:idx_end, :);
                Y_i_n = Y_final_n(idx_start:idx_end, :);
            end
    
            % Extract corresponding block Omega and Y
            Omega_i = Omega_final(idx_start:idx_end, :);
            Y_i = Y_final(idx_start:idx_end, :);

            % Partition into s and r
            num_skele = k;
            node(nd).Indices.skeleton  = node(nd).Indices.original(1:num_skele);
            node(nd).Indices.redundant = node(nd).Indices.original(num_skele+1:end);

        elseif dim == 2
            % ========================= 2D =========================
            blk_idx = nd - start_idx;
            row_idx = mod(blk_idx, num_blocks_side);
            col_idx = floor(blk_idx / num_blocks_side);
    
            row_start = row_idx * pts_per_block + 1;
            row_end   = (row_idx + 1) * pts_per_block;
            col_start = col_idx * pts_per_block + 1;
            col_end   = (col_idx + 1) * pts_per_block;
    
            [cols_grid, rows_grid] = meshgrid(col_start:col_end, row_start:row_end);
            indices = sub2ind([n1d, n1d], rows_grid(:), cols_grid(:))';
    
            if level == num_levels
                node(nd).Indices.original = indices;
            else
                num_parents_side = 2^(level-1);  % e.g., level 2: 2 parents/side (for 4 total)
    
                % The block ID (0-based) for this parent in the grid of parents
                parent_block_id = nd - start_idx; % 0-based
                parent_row = mod(parent_block_id, num_parents_side);
                parent_col = floor(parent_block_id / num_parents_side);
    
                % Info for the children level
                child_level = level + 1;
                start_child_idx = sum(4.^(0:child_level-2)) + 1;
                num_children_side = 2^(child_level-1); % number of blocks/side on children level
    
                % Find the 4 children by their (row, col) in the finer grid
                % The formula maps (row,col) to a 1D index (column-major block order)
                child_blocks = [ ...
                    2*parent_row     + (2*parent_col)    *num_children_side,   % NW
                    2*parent_row + 1 + (2*parent_col)    *num_children_side,   % SW
                    2*parent_row     + (2*parent_col + 1)*num_children_side,   % NE
                    2*parent_row + 1 + (2*parent_col + 1)*num_children_side];  % SE
    
                child_indices = start_child_idx + child_blocks;
            
                % --- Merge children's skeleton indices for this parent ---
                total_len = 0; % to preallocate
                for ii = 1:length(child_indices)
                    cid = child_indices(ii);
                    total_len = total_len + numel(node(cid).Indices.skeleton);
                end
    
                original_indices = zeros(1, total_len);
    
                offset = 0;
                for ii = 1:length(child_indices)
                    cid = child_indices(ii);
                    skel = node(cid).Indices.skeleton;
                    n = numel(skel);
                    original_indices(offset + 1 : offset + n) = skel; % not sorted
                    offset = offset + n;
                end
            
                node(nd).Indices.original = original_indices;
            end
    
            idx_block = node(nd).Indices.original;
    
            if level < num_levels
                %-------------------------------
                % INDEX MAPPING
                %-------------------------------
    
                [~, locs] = ismember(idx_block(:), all_skel);
                local_idx = locs;  % these are the row‐positions in Omega_final/Y_final
    
                if ~ismember(kernel_type, symmetric_kernel_list)
                    Omega_i_n = Omega_final_n(local_idx, :);
                    Y_i_n     = Y_final_n(local_idx,     :);
                end
    
                Omega_i = Omega_final(local_idx, :);
                Y_i     = Y_final(local_idx,     :);
            else
                % Leaf: can index directly
                if ~ismember(kernel_type, symmetric_kernel_list)
                    Omega_i_n = Omega_final_n(idx_block, :);
                    Y_i_n     = Y_final_n(idx_block,     :);
                end
    
                Omega_i = Omega_final(idx_block, :);
                Y_i     = Y_final(idx_block,     :);
            end

            % Partition into s and r
            num_skele = k;
            node(nd).Indices.skeleton  = idx_block(1:num_skele);
            node(nd).Indices.redundant = idx_block(num_skele+1:end);
        end

        U_struct = U_full{blk_idx + 1};
        node(nd).U = U_struct; % Store the struct {V, tau}

        node(nd).isLeaf = (level == num_levels);

        if ismember(kernel_type, symmetric_kernel_list)
            % Y_i_tilde = U_i' * Y_i;
            Y_i_tilde = apply_householder(U_struct.W, U_struct.C, Y_i, 'transpose');

            if ismember(kernel_type, {'helmholtz'})
                % Omega_i_tilde = U_i.' * Omega_i = conj(U' * conj(Omega))
                Omega_conj = conj(Omega_i);
                temp_om = apply_householder(U_struct.W, U_struct.C, Omega_conj, 'transpose');
                Omega_i_tilde = conj(temp_om);
            else 
                % Omega_i_tilde = U_i' * Omega_i;
                Omega_i_tilde = apply_householder(U_struct.W, U_struct.C, Omega_i, 'transpose');
            end

        else
            V_struct = V_full{blk_idx + 1};
            node(nd).V = V_struct;
    
            % Apply Y_i_tilde_n and Omega_i_tilde_n
            Y_i_tilde_n = apply_householder(V_struct.W, V_struct.C, Y_i_n, 'transpose');
            Omega_i_tilde_n = apply_householder(U_struct.W, U_struct.C, Omega_i_n, 'transpose');

            node(nd).Y_tilde_n = Y_i_tilde_n;
            node(nd).Omega_tilde_n = Omega_i_tilde_n(1:k, :);

            % Apply Y_i_tilde and Omega_i_tilde
            Y_i_tilde = apply_householder(U_struct.W, U_struct.C, Y_i, 'transpose');
            Omega_i_tilde = apply_householder(V_struct.W, V_struct.C, Omega_i, 'transpose');
        end

        node(nd).Y_tilde = Y_i_tilde;
        node(nd).Omega_tilde = Omega_i_tilde(1:k, :);

        node(nd).rank = k;
    
        % Solve for X_rs, X_rr (and X_sr)
        nodeIndices_i = node(nd).Indices;

        [X_rs_i, X_rr_i] = solve_Xrsrr(Y_i_tilde, Omega_i_tilde, nodeIndices_i, kernel_type, symmetric_kernel_list, spd_kernel_list, level);

        node(nd).X_rs = X_rs_i;
        node(nd).X_rr = X_rr_i;
        if ismember(kernel_type, {'helmholtz'})
            node(nd).X_srX_rr = xrr_solve(X_rr_i, X_rs_i.', 'right');
        else
            node(nd).X_srX_rr = xrr_solve(X_rr_i, X_rs_i', 'right');
        end

        if ~ismember(kernel_type, symmetric_kernel_list)
            [X_sr_i_T, ~] = solve_Xrsrr(Y_i_tilde_n, Omega_i_tilde_n, nodeIndices_i, kernel_type, symmetric_kernel_list, spd_kernel_list, level);
            X_sr_i = X_sr_i_T';
            node(nd).X_srX_rr = xrr_solve(X_rr_i, X_sr_i, 'right');
            node(nd).X_rsX_rr = xrr_solve(X_rr_i', X_rs_i', 'right');
        end
    end
end