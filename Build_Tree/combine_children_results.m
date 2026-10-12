function [Y_i_hat_3, Omega_i_hat_3, Y_i_hat_3_n, Omega_i_hat_3_n] = combine_children_results(children, kernel_type, kernel_list, index_or_step_b)
    % combineChildrenResults: Combine results from 2 child nodes in 1D HSS tree.
    %
    % Usage:
    %   % Full combine (all columns)
    %   [Yh, Oh, Yhn, Ohn] = combineChildrenResults(children, 'laplace', [], {'laplace','poisson','gaussian','helmholtz'});
    %
    %   % Extra: last step_b columns
    %   [Yh, Oh, Yhn, Ohn] = combineChildrenResults(children, 'gaussian', 5, {'laplace','poisson','gaussian','helmholtz'});
    %
    %   % Extra: specify column indices
    %   [Yh, Oh, Yhn, Ohn] = combineChildrenResults(children, 'poisson', 5:10, {'laplace','poisson','gaussian','helmholtz'});
    %
    % Inputs:
    %   children             - Array of child node structs
    %   kernel_type          - Kernel type
    %   kernel_list          - Cell array of kernels to decide if 
    %                          kernel type is symmetric/unsymmetric  
    %   index_or_step_b      - (optional) Index of columns to reuse or
    %                          Number of extra columns appended at the leaf
    %
    % Outputs:
    %   Y_i_hat_3, Omega_i_hat_3
    %   Y_i_hat_3_n, Omega_i_hat_3_n (empty if kernel_type in kernel_list)

    % ---- Determine symmetry ----
    is_unsymmetric = ~ismember(kernel_type, kernel_list);

    % ---- Determine column selection ----
    total_cols = size(children(1).Y_tilde, 2);

    if nargin < 4 || isempty(index_or_step_b) 
        cols = 1:total_cols;
        if is_unsymmetric
            total_cols_n = size(children(1).Y_tilde_n, 2);
            cols_n = 1:total_cols_n;
        end
    elseif isscalar(index_or_step_b)
        step_b = index_or_step_b;
        cols = (total_cols - step_b + 1) : total_cols;

        if is_unsymmetric
            total_cols_n = size(children(1).Y_tilde_n, 2);
            cols_n = (total_cols_n - step_b + 1) : total_cols_n;
        end
    else
        cols = index_or_step_b;
        cols_n = index_or_step_b;
    end

    % ---- Preallocate outputs ----
    num_skele_total = 0;
    for i = 1:length(children)
        num_skele_total = num_skele_total + length(children(i).Indices.skeleton);
    end

    Y_i_hat_3 = zeros(num_skele_total, length(cols), ...
        'like', children(1).Y_tilde);
    Omega_i_hat_3 = zeros(num_skele_total, length(cols), ...
        'like', children(1).Omega_tilde);

    if is_unsymmetric
        Y_i_hat_3_n = zeros(num_skele_total, length(cols_n), ...
            'like', children(1).Y_tilde_n);
        Omega_i_hat_3_n = zeros(num_skele_total, length(cols_n), ...
            'like', children(1).Omega_tilde_n);
    else
        Y_i_hat_3_n = [];
        Omega_i_hat_3_n = [];
    end

    % ---- Assemble results ----
    current_row = 1;
    for i = 1:length(children)
        node_child = children(i);
        num_skele = length(node_child.Indices.skeleton);
        row_range = current_row:(current_row + num_skele - 1);
        
        X_srX_rr = node_child.X_srX_rr;
        Y_tilde = node_child.Y_tilde(:, cols);
        Omega_tilde = node_child.Omega_tilde(:, cols);

        Y_i_hat_3(row_range, :) = Y_tilde(1:num_skele,:) -  X_srX_rr * Y_tilde(num_skele+1:size(Y_tilde,1),:);
        Omega_i_hat_3(row_range, :) = Omega_tilde(1:num_skele,:);

        if is_unsymmetric
            X_rsX_rr = node_child.X_rsX_rr;
            Y_tilde_n = node_child.Y_tilde_n(:, cols_n);
            Omega_tilde_n = node_child.Omega_tilde_n(:, cols_n);
        
            Y_i_hat_3_n(row_range, :) = Y_tilde_n(1:num_skele,:) - X_rsX_rr * Y_tilde_n(num_skele+1:end,:);
            Omega_i_hat_3_n(row_range, :) = Omega_tilde_n(1:num_skele,:);
        end

        current_row = current_row + num_skele;
    end
end
