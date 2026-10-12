function print_summary_table(table_data)
    fprintf('%-8s %-7s %-7s %-12s %-12s %-10s %-10s %-12s %-10s %-13s %-11s %-14s %-6s %-16s %-9s\n', ...
            'tol', 'level','n','num_blocks','block_size','chosen_k', ...
            'n_mv', 't_mv (s)', 't_fact (s)', 't_solve (s)', 'residual', ...
            'factor (MB)','n_it', 'relres', 'n_it_np');

    for i = 1:size(table_data,1)
        tol_str       = safe_str(table_data{i,1});
        level_str     = safe_num(table_data{i,2});
        n_str         = safe_num(table_data{i,3});
        nb_str        = safe_num(table_data{i,4});
        bs_str        = safe_num(table_data{i,5});
        k_str         = safe_num(table_data{i,6});
        nmv_str       = safe_num(table_data{i,7});
        tmv_str       = safe_str(table_data{i,8});
        tfact_str     = safe_str(table_data{i,9});
        tsolve_str    = safe_str(table_data{i,10});
        res_str       = safe_str(table_data{i,11});
        space_str     = safe_str(table_data{i,12});
        nit_str       = safe_num(table_data{i,13});
        relres_str    = safe_str(table_data{i,14});
        nit_np_str    = safe_num(table_data{i,15});
    
        fprintf('%-8s %-7s %-7s %-12s %-12s %-10s %-10s %-12s %-10s %-13s %-11s %-14s %-6s %-16s %-9s\n', ...
            tol_str, level_str, n_str, nb_str, bs_str, k_str, ...
            nmv_str, tmv_str, tfact_str, tsolve_str, res_str, ...
            space_str, nit_str, relres_str, nit_np_str);
    end
end

function out = safe_num(val)
    if isempty(val) || (isnumeric(val) && val == 0) || (isstring(val) && val == "&")
        out = '';
    else
        out = num2str(val);
    end
end

function out = safe_str(val)
    if isempty(val) || (isnumeric(val) && val == 0) || (isstring(val) && val == "&")
        out = '';
    else
        out = string(val);
    end
end
