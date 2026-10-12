function Ax_recovered = mat_vec(node, X, num_levels, kernel_type, dim)
    % matVec: Perform matrix multiplication using an upward and 
    % downward tree traversal.

    % ---- Upward tree multiplication ----
    % Y = ...*I_P_hat*I_L_hat^{-1}*I_V_hat_T*P*L^{-1}*V_T*b
    Y = X;
    
    for level = num_levels:-1:2
        if dim == 1
            start_node_idx = 2^(level-1);    % Example: level=3 -> start_node_idx=4
            end_node_idx   = 2^level - 1;    % Example: level=3 -> end_node_idx=7
        elseif dim == 2
            start_node_idx = sum(4.^(0:level-2)) + 1;
            end_node_idx = sum(4.^(0:level-1));
        end 
    
        for nd = start_node_idx : end_node_idx
            % Extract node information
            X_rr_i_uw  = node(nd).X_rr;
            X_rs_i_uw  = node(nd).X_rs;
    
            % U_T * X
            if ismember(kernel_type, {'laplace', 'poisson_schur', ...
                    'helmholtz_schur', 'matern'})
                U_struct = node(nd).U;
                Y(node(nd).Indices.original, :) = apply_householder(U_struct.W, U_struct.C, Y(node(nd).Indices.original, :), 'transpose');
            elseif ismember(kernel_type, {'helmholtz'})
                % U.' * Y = conj(U' * conj(Y))
                U_struct = node(nd).U;
                y_conj = conj(Y(node(nd).Indices.original, :));
                temp = apply_householder(U_struct.W, U_struct.C, y_conj, 'transpose');
                Y(node(nd).Indices.original, :) = conj(temp);
            else
                V_struct = node(nd).V;
                Y(node(nd).Indices.original, :) = apply_householder(V_struct.W, V_struct.C, Y(node(nd).Indices.original, :), 'transpose');
            end
        
            % L^{-1} * U_T * X
            Y(node(nd).Indices.redundant, :) = xrr_solve(X_rr_i_uw, X_rs_i_uw * Y(node(nd).Indices.skeleton, :), 'left') ...
                                                + Y(node(nd).Indices.redundant, :);
        end
    end 

    % ---- Downward tree multiplication ----
    % Ax_recovered = U*R^{-1}*P_T*I_U_hat*I_R_hat^{-1}*I_P_hat_T*...*A_hat*Y
    Ax_recovered = Y;
    
    % Process root node
    Ax_recovered(node(1).Indices.redundant, :) = node(1).X_rr * ...
        Ax_recovered(node(1).Indices.redundant, :);
    
    % Process non-root nodes
    for level = 2:num_levels
         if dim == 1
            start_node_idx = 2^(level-1);
            end_node_idx   = 2^level - 1;
         elseif dim == 2
            start_node_idx = sum(4.^(0:level-2)) + 1;
            end_node_idx = sum(4.^(0:level-1));
         end 
     
         for nd = start_node_idx : end_node_idx
            % Extract node information
            U_struct     = node(nd).U;
            X_rr_i_dw  = node(nd).X_rr;
            X_srX_rr_i_dw  = node(nd).X_srX_rr;
    
            % A_hat * y
            Ax_recovered(node(nd).Indices.redundant, :) = X_rr_i_dw * ...
                Ax_recovered(node(nd).Indices.redundant, :);
    
            % R^{-1} * A_hat * y
            Ax_recovered(node(nd).Indices.skeleton, :) = Ax_recovered(node(nd).Indices.skeleton, :) ...
                            + X_srX_rr_i_dw * Ax_recovered(node(nd).Indices.redundant, :);
    
            % U * R^{-1} * A_hat * y
            Ax_recovered(node(nd).Indices.original, :) = apply_householder(U_struct.W, U_struct.C, ...
                                                          Ax_recovered(node(nd).Indices.original, :), 'normal');
         end
    end
end
