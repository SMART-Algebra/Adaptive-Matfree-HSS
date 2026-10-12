function x_recovered = linear_solve(node, b, num_levels, kernel_type, dim)
    % linearSolve: Solve a system of equations using upward and 
    % downward tree traversals.

    % ---- Upward tree solve: y = ...*I_P_hat*I_R_hat*I_U_hat_T*P*R*U_T*b ----
    y = b;
    
    for level = num_levels:-1:2
        if dim ==  1
            start_node_idx = 2^(level-1);    % Example: level=3 -> start_node_idx=4
            end_node_idx   = 2^level - 1;    % Example: level=3 -> end_node_idx=7
        elseif dim == 2
            start_node_idx = sum(4.^(0:level-2)) + 1;
            end_node_idx = sum(4.^(0:level-1));
        end
    
        for nd = start_node_idx : end_node_idx
            % Extract node information
            U_struct = node(nd).U;
            X_srX_rr_i_uw  = node(nd).X_srX_rr;

            indices_orig = node(nd).Indices.original;
            indices_skele = node(nd).Indices.skeleton;
            indices_redun = node(nd).Indices.redundant;
    
            % U_T * b
            y(indices_orig, :) = apply_householder(U_struct.W, U_struct.C, y(indices_orig, :), 'transpose');
        
            % R * U_T * b
            y(indices_skele, :) = y(indices_skele, :) - X_srX_rr_i_uw * y(indices_redun, :);
        end
    end 

    % ---- Downward tree solve: x_recovered = V*L*P_T*I_V_hat*I_L_hat*I_P_hat_T*...*A_hat\y ----
    x_recovered = y;
    
    % -- Process root node --
    x_recovered(node(1).Indices.redundant, :) = xrr_solve(node(1).X_rr, ...
        x_recovered(node(1).Indices.redundant, :), 'left');
    
    % -- Process non-root nodes --
    for level = 2:num_levels
        if dim ==  1
            start_node_idx = 2^(level-1);    % Example: level=3 -> start_node_idx=4
            end_node_idx   = 2^level - 1;    % Example: level=3 -> end_node_idx=7
        elseif dim == 2
            start_node_idx = sum(4.^(0:level-2)) + 1;
            end_node_idx = sum(4.^(0:level-1));
        end
     
        for nd = start_node_idx : end_node_idx
            % Extract node information
            X_rr_i_dw  = node(nd).X_rr;
            X_rs_i_dw  = node(nd).X_rs;

            indices_orig = node(nd).Indices.original;
            indices_skele = node(nd).Indices.skeleton;
            indices_redun = node(nd).Indices.redundant;
    
            % L * A_hat^{-1} * y
            rhs_vec = x_recovered(indices_redun, :) - X_rs_i_dw * x_recovered(indices_skele, :);
            x_recovered(indices_redun, :) = xrr_solve(X_rr_i_dw, rhs_vec, 'left');
    
            % U * L * A_hat^{-1} * y
            if ismember(kernel_type, {'laplace', 'poisson_schur', 'helmholtz_schur', 'gaussian', 'matern'})
                U_struct = node(nd).U;
                x_recovered(indices_orig, :) = apply_householder(U_struct.W, U_struct.C, x_recovered(indices_orig, :), 'normal');
            elseif ismember(kernel_type, {'helmholtz'})
                % conj(U) * x = conj(U * conj(x))
                U_struct = node(nd).U;
                
                x_conj = conj(x_recovered(indices_orig, :));
                temp = apply_householder(U_struct.W, U_struct.C, x_conj, 'normal');
                x_recovered(indices_orig, :) = conj(temp);
            else
                V_struct = node(nd).V;
                x_recovered(indices_orig, :) = apply_householder(V_struct.W, V_struct.C, x_recovered(indices_orig, :), 'normal');
            end

        end
    end
end
