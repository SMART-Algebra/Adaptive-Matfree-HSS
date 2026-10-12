function node = process_node(node, nd, Y_i, Omega_i, k_A, isLeaf, kernel_type, symmetric_kernel_list, spd_kernel_list, Y_n_i, Omega_n_i)
    % processNode updates node(nd) with Householder U (W, C), Y_tilde(_n),
    % Omega_tilde(_n), skeleton/redundant indices, X_rs, X_rr, X_sr

    if nargin < 10
        is_unsymmetric = false;
    else
        is_unsymmetric = true;
    end

    [W_i, C_i] = U_function(k_A, Y_i, Omega_i);

    node(nd).U.W = W_i;
    node(nd).U.C = C_i;
    node(nd).isLeaf = isLeaf;

    if is_unsymmetric
        [W_vi, C_vi] = U_function(k_A, Y_n_i, Omega_n_i);
        node(nd).V.W = W_vi;
        node(nd).V.C = C_vi;

        Y_i_tilde_n = apply_householder(W_vi, C_vi, Y_n_i, 'transpose');
        Omega_i_tilde_n = apply_householder(W_i, C_i, Omega_n_i, 'transpose');
    
        node(nd).Omega_tilde_n = Omega_i_tilde_n;
        node(nd).Y_tilde_n = Y_i_tilde_n;

        Y_i_tilde = apply_householder(W_i, C_i, Y_i, 'transpose');
        Omega_i_tilde = apply_householder(W_vi, C_vi, Omega_i, 'transpose');
    else
        Y_i_tilde = apply_householder(W_i, C_i, Y_i, 'transpose');

        % Omega_i_tilde = U_i.' * Omega_i;
        Omega_conj = conj(Omega_i);
        temp_om = apply_householder(W_i, C_i, Omega_conj, 'transpose');
        Omega_i_tilde = conj(temp_om);
    end

    node(nd).Y_tilde = Y_i_tilde;
    node(nd).Omega_tilde = Omega_i_tilde;

    % Partition into s and r
    num_skele = k_A;
    node(nd).Indices.skeleton  = node(nd).Indices.original(1:num_skele);
    node(nd).Indices.redundant = node(nd).Indices.original(num_skele+1:end);

    % Solve for X_rs, X_rr
    nodeIndices_i = node(nd).Indices;
    
    [X_rs_i, X_rr_i] = solve_Xrsrr(Y_i_tilde, Omega_i_tilde, nodeIndices_i, kernel_type, symmetric_kernel_list, spd_kernel_list);

    node(nd).X_srX_rr = xrr_solve(X_rr_i, X_rs_i.', 'right');
    node(nd).X_rr = X_rr_i;
    node(nd).X_rs = X_rs_i;

    if is_unsymmetric
        [X_sr_i, ~] = solve_Xrsrr(Y_i_tilde_n, Omega_i_tilde_n, nodeIndices_i, kernel_type, symmetric_kernel_list, spd_kernel_list);
        node(nd).X_sr = X_sr_i';
    end
end
