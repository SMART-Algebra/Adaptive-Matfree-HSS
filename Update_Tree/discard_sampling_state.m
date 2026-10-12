function node = discard_sampling_state(node)
%DISCARD_SAMPLING_STATE Remove construction-only samples from an HSS tree.
%   The transformed samples are required while moving up the tree, but the
%   completed factorization and its application use only the bases,
%   elimination factors, and index sets. Releasing these fields reduces the
%   live memory retained during solves without changing factor storage.

    sample_fields = { ...
        'Y_tilde', 'Omega_tilde', 'Y_tilde_n', 'Omega_tilde_n'};
    for node_id = 1:numel(node)
        for field_id = 1:numel(sample_fields)
            name = sample_fields{field_id};
            if isfield(node, name)
                node(node_id).(name) = [];
            end
        end
    end
end
