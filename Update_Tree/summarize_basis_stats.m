function diagnostics = summarize_basis_stats(method, level_stats)
%SUMMARIZE_BASIS_STATS Aggregate diagnostics from adaptive basis construction.

    diagnostics = struct( ...
        'method', char(method), ...
        'level_stats', level_stats, ...
        'nullspace_update_time', 0, ...
        'diagnostics_time', 0, ...
        'max_nullspace_residual', NaN, ...
        'max_orthogonality_error', NaN, ...
        'min_initial_rcond', NaN, ...
        'safeguard_fallback_count', 0);

    if isempty(level_stats)
        return;
    end

    diagnostics.nullspace_update_time = sum([level_stats.nullspace_time]);
    diagnostics.diagnostics_time = sum([level_stats.diagnostics_time]);
    diagnostics.safeguard_fallback_count = ...
        sum([level_stats.safeguard_fallback_count]);

    values = [level_stats.max_nullspace_residual];
    values = values(isfinite(values));
    if ~isempty(values)
        diagnostics.max_nullspace_residual = max(values);
    end

    values = [level_stats.max_orthogonality_error];
    values = values(isfinite(values));
    if ~isempty(values)
        diagnostics.max_orthogonality_error = max(values);
    end

    values = [level_stats.min_initial_rcond];
    values = values(isfinite(values));
    if ~isempty(values)
        diagnostics.min_initial_rcond = min(values);
    end
end
