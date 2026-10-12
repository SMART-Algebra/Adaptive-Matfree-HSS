function [Y_extra_all, Omega_extra_all, node] = expand_Omega_Y(node, current_level, index_or_step_b, mv, n, fixed_pt_leaf, kernel_type, symmetric_kernel_list, dim, sample_distribution)
    % expandOmegaY: Appends step_b to Omega and Y at a current
    % non-leaf level by either:
    %   - Adding new random columns at leaf level and propagating upward (scalar step_b), OR
    %   - Reusing existing specified columns (vector cols).
    % Shows up in: building_bases.m

    if nargin < 10 || isempty(sample_distribution)
        sample_distribution = 'real';
    end

    if dim == 1
        leaf_level = log2(length(node)+1);
    elseif dim == 2
        leaf_level = log(length(node)*(4-1)+1)/log(4);
    end

    if isscalar(index_or_step_b)
        step_b = index_or_step_b;

        % --- Add columns to leaf nodes (one-time, shared for all upward propagation) ---
        [new_cols_Omega_leaf, new_cols_Y_leaf] = add_columns_at_leaves(step_b, mv, n, sample_distribution);

        % --- Update all nodes at levels up to current_level - 1 ---

        for lvl = leaf_level:-1:(current_level+1)
            if dim == 1
                lvl_start = 2^(lvl-1);
                lvl_end = 2^lvl - 1;
            elseif dim == 2
                lvl_start = sum(4.^(0:lvl-2)) + 1;
                lvl_end = sum(4.^(0:lvl-1));
            end
            
            for nd = lvl_start:lvl_end
                if lvl == leaf_level
                    if dim == 1
                        % ========================= 1D =========================
                        blk_idx = nd - lvl_start + 1;
                        row_start = (blk_idx-1)*fixed_pt_leaf + 1;
                        row_end = blk_idx*fixed_pt_leaf;
                        indices = row_start:row_end;

                    elseif dim == 2
                        % ========================= 2D =========================
                        num_blocks_side = 2^(lvl-1);
                        block_size_side = sqrt(fixed_pt_leaf);
    
                        blk_idx = nd - lvl_start;
                        row_idx = mod(blk_idx, num_blocks_side);
                        col_idx = floor(blk_idx / num_blocks_side);
    
                        row_start = row_idx * block_size_side + 1;
                        row_end = (row_idx + 1) * block_size_side;
                        col_start = col_idx * block_size_side + 1;
                        col_end = (col_idx + 1) * block_size_side;
            
                        [cols_grid, rows_grid] = meshgrid(col_start:col_end, row_start:row_end);
                        indices = sub2ind([sqrt(n), sqrt(n)], rows_grid(:), cols_grid(:))';
                    end

                    new_Y_block = new_cols_Y_leaf(indices, :);
                    new_Omega_block = new_cols_Omega_leaf(indices, :);

                else 
                    if dim == 1
                        % ========================= 1D =========================
                        left_child_idx = 2 * nd;
                        right_child_idx = 2 * nd + 1;
    
                        child_indices = [left_child_idx, right_child_idx];

                    elseif dim == 2
                        % ========================= 2D =========================
                        num_parents_side = 2^(lvl-1);  % e.g., level 2: 2 parents/side (for 4 total)
            
                        % The block ID (0-based) for this parent in the grid of parents
                        parent_block_id = nd - lvl_start; % 0-based
                        parent_row = mod(parent_block_id, num_parents_side);
                        parent_col = floor(parent_block_id / num_parents_side);
            
                        % Info for the children level
                        child_level = lvl + 1;
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
                    end

                    [new_Y_block, new_Omega_block, new_Y_block_n, new_Omega_block_n] = combine_children_results(node(child_indices), ...
                                                                                            kernel_type, symmetric_kernel_list, step_b);

                    if strcmp(kernel_type, 'VU')
                        new_Y_block = new_Y_block_n;
                        new_Omega_block = new_Omega_block_n;
                    end
                end

                % Update tilde
                U_struct = node(nd).U;
                k = node(nd).rank;

                if ismember(kernel_type, symmetric_kernel_list)
                    if ismember(kernel_type, {'helmholtz'})
                        % U' * Y
                        % U.' * Omega = conj(U' * conj(Omega))
                        % node(nd).Y_tilde = [node(nd).Y_tilde, U' * new_Y_block];
                        % node(nd).Omega_tilde = [node(nd).Omega_tilde, U_transpose(1:k,:) * new_Omega_block];

                        Y_proj = apply_householder(U_struct.W, U_struct.C, new_Y_block, 'transpose');
                        node(nd).Y_tilde = [node(nd).Y_tilde, Y_proj];
                        
                        Omega_conj = conj(new_Omega_block);
                        Proj = U_struct.W' * Omega_conj;
                        Omega_top_k = Omega_conj(1:k, :) - U_struct.C(1:k, :) * Proj;
                        node(nd).Omega_tilde = [node(nd).Omega_tilde, conj(Omega_top_k)];
                    else
                        Y_proj = apply_householder(U_struct.W, U_struct.C, new_Y_block, 'transpose');
                        node(nd).Y_tilde = [node(nd).Y_tilde, Y_proj];

                       % Omega_tilde = U' * new_Omega_block
                        Proj = U_struct.W' * new_Omega_block;
                        Omega_top_k = new_Omega_block(1:k, :) - U_struct.C(1:k, :) * Proj;
                        node(nd).Omega_tilde = [node(nd).Omega_tilde, Omega_top_k];
                    end
                elseif strcmp(kernel_type, 'UV')
                    V_struct = node(nd).V;
                    
                    % Y_tilde = U' * new_Y_block
                    Y_proj = apply_householder(U_struct.W, U_struct.C, new_Y_block, 'transpose');
                    node(nd).Y_tilde = [node(nd).Y_tilde, Y_proj];
                    
                    % Omega_tilde = V' * new_Omega_block
                    Proj = V_struct.W' * new_Omega_block;
                    Omega_top_k = new_Omega_block(1:k, :) - V_struct.C(1:k, :) * Proj;
                    node(nd).Omega_tilde = [node(nd).Omega_tilde, Omega_top_k];
                else
                    V_struct = node(nd).V;
                    
                    % Y_tilde_n = V' * new_Y_block
                    Y_proj_n = apply_householder(V_struct.W, V_struct.C, new_Y_block, 'transpose');
                    node(nd).Y_tilde_n = [node(nd).Y_tilde_n, Y_proj_n];
                    
                    % Omega_tilde_n = U' * new_Omega_block
                    Proj = U_struct.W' * new_Omega_block;
                    Omega_top_k = new_Omega_block(1:k, :) - U_struct.C(1:k, :) * Proj;
                    node(nd).Omega_tilde = [node(nd).Omega_tilde, Omega_top_k];
                end
             end
        end
    %     cols_for_parent = step_b;
    % else
    %     cols_for_parent = index_or_step_b; % reuse provided column indices
    end

    if dim == 1
        start_node_idx = 2^(current_level-1);
        end_node_idx = 2^current_level - 1;
    elseif dim == 2
        start_node_idx = sum(4.^(0:current_level-2)) + 1;
        end_node_idx = sum(4.^(0:current_level-1));
    end
    
    num_nodes = end_node_idx - start_node_idx + 1;

    Y_extra_cell = cell(num_nodes, 1); % the current level
    Omega_extra_cell = cell(num_nodes, 1);

    for current_idx = start_node_idx:end_node_idx
        if dim == 1
        % ========================= 1D =========================
            left_child_idx = 2 * current_idx;
            right_child_idx = 2 * current_idx + 1;
            child_indices = [left_child_idx, right_child_idx];

        elseif dim == 2
        % ========================= 1D =========================
            num_parents_side = 2^(current_level-1);  % e.g., level 2: 2 parents/side (for 4 total)
    
            % The block ID (0-based) for this parent in the grid of parents
            parent_block_id = current_idx - start_node_idx; % 0-based
            parent_row = mod(parent_block_id, num_parents_side);
            parent_col = floor(parent_block_id / num_parents_side);
    
            % Info for the children level
            child_level = current_level + 1;
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
        end

        [Y_i_hat_3_extra, Omega_i_hat_3_extra, Y_i_hat_3_extra_n, Omega_i_hat_3_extra_n] = combine_children_results(node(child_indices), ...
                                                                                                kernel_type, symmetric_kernel_list, index_or_step_b);

        if ismember(kernel_type, [symmetric_kernel_list, 'UV'])
            Y_extra_cell{current_idx - start_node_idx + 1} = Y_i_hat_3_extra;
            Omega_extra_cell{current_idx - start_node_idx + 1} = Omega_i_hat_3_extra;
        else
            Y_extra_cell{current_idx - start_node_idx + 1} = Y_i_hat_3_extra_n;
            Omega_extra_cell{current_idx - start_node_idx + 1} = Omega_i_hat_3_extra_n;
        end
    end

    Y_extra_all = vertcat(Y_extra_cell{:});
    Omega_extra_all = vertcat(Omega_extra_cell{:});
end
