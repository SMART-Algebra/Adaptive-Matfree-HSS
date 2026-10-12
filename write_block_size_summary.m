function S = write_block_size_summary(raw, output_dir)
%WRITE_BLOCK_SIZE_SUMMARY Median over seeds of the quantities the paper tables report.
%   One row per (case, b, pruning). Written so the tables can be built
%   without transferring the full per-seed file.
    T = struct2table(raw);
    [G, cname, bb, sc] = findgroups(string(T.case_name), T.step_b, ...
        string(T.rank_pruning));
    S = table(cname, bb, sc, 'VariableNames', ...
        {'case_name', 'step_b', 'rank_pruning'});
    S.n_seeds       = splitapply(@numel,  T.final_rank,     G);
    S.rank_med      = splitapply(@median, T.final_rank,     G);
    S.memory_mb_med = splitapply(@median, T.memory_mb,      G);
    S.t_factor_med  = splitapply(@median, T.t_factor,       G);
    S.residual_med  = splitapply(@median, T.solve_residual, G);
    writetable(S, fullfile(output_dir, 'block_size_summary.csv'));
end
