function level_stats = append_basis_stat(level_stats, basis_stat, basis_name)
%APPEND_BASIS_STAT Append a labeled U- or V-basis diagnostic record.
    basis_stat.basis = char(basis_name);
    if isempty(level_stats)
        level_stats = basis_stat;
    else
        level_stats(end+1) = basis_stat;
    end
end
