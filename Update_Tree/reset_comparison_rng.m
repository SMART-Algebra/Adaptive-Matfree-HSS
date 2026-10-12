function reset_comparison_rng(base_seed, level, basis_name)
%RESET_COMPARISON_RNG Synchronize fresh samples at each level for comparisons.
%   No action is taken when base_seed is empty, preserving normal runs.

    if isempty(base_seed)
        return;
    end

    basis_offset = 0;
    if strcmpi(basis_name, 'V')
        basis_offset = 100000;
    end
    rng(double(base_seed) + 1000 * double(level) + basis_offset, 'twister');
end
