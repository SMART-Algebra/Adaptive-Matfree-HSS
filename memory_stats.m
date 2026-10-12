function factor_storage_mb = memory_stats(node)
%MEMORY_STATS Storage of the completed factorized representation in MiB.
%   Construction-only transformed samples are excluded. This is an
%   algorithmic storage metric, not the peak resident memory of MATLAB;
%   record Slurm MaxRSS separately when peak process memory is needed.

    node_info = whos('node');
    total_node_bytes = node_info.bytes;

    excluded_fields = { ...
        'Y_tilde', 'Omega_tilde', 'Y_tilde_n', 'Omega_tilde_n', ...
        'isLeaf', 'rank'};
    excluded_bytes = 0;
    for field_id = 1:numel(excluded_fields)
        name = excluded_fields{field_id};
        if ~isfield(node, name)
            continue;
        end
        for node_id = 1:numel(node)
            value = node(node_id).(name); %#ok<NASGU>
            value_info = whos('value');
            excluded_bytes = excluded_bytes + value_info.bytes;
        end
    end

    factor_storage_mb = round( ...
        (total_node_bytes - excluded_bytes) / 1024^2, 2);
end
