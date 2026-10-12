function [Y_current_all, Omega_current_all, Y_current_all_n, Omega_current_all_n] = get_YOmega_non_leaf(node, current_level, ...
    kernel_type, symmetric_kernel_list, dim)
    % getYOmega_nonLeaf: Obtains the full Y, Omega, Y_n and Omega_n for the current level

    if dim == 1
        start_node_idx = 2^(current_level-1);
        end_node_idx = 2^current_level - 1;
    elseif dim == 2
        start_node_idx = sum(4.^(0:current_level-2)) + 1;
        end_node_idx = sum(4.^(0:current_level-1));
    end

    num_nodes = end_node_idx - start_node_idx + 1;
    
    Y_cell = cell(num_nodes, 1);
    Omega_cell = cell(num_nodes, 1);
    Y_cell_n = cell(num_nodes, 1);
    Omega_cell_n = cell(num_nodes, 1);

    for current_idx = start_node_idx:end_node_idx
        if dim == 1
            % ========================= 1D =========================
            left_child_idx = 2 * current_idx;
            right_child_idx = 2 * current_idx + 1;

            child_indices = [left_child_idx, right_child_idx];
            
        elseif dim == 2
            % ========================= 2D =========================
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

        [Y_i_hat_3, Omega_i_hat_3, Y_i_hat_3_n, Omega_i_hat_3_n] = combine_children_results(node(child_indices), kernel_type, symmetric_kernel_list, []);
        
        idx = current_idx - start_node_idx + 1;

        Y_cell{idx} = Y_i_hat_3;
        Omega_cell{idx} = Omega_i_hat_3;
        
        if ~ismember(kernel_type, symmetric_kernel_list)
            Y_cell_n{idx} = Y_i_hat_3_n;
            Omega_cell_n{idx} = Omega_i_hat_3_n;
        end
    end

    % Vertically stack the extra columns from all parent nodes.
    Y_current_all = vertcat(Y_cell{:});
    Omega_current_all = vertcat(Omega_cell{:});

    if ismember(kernel_type, symmetric_kernel_list)
        Y_current_all_n = [];
        Omega_current_all_n = [];
    else
        Y_current_all_n = vertcat(Y_cell_n{:});
        Omega_current_all_n = vertcat(Omega_cell_n{:});
    end
end
