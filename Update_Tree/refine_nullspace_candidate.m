function [candidate, residual] = refine_nullspace_candidate( ...
        Omega, candidate, retained_basis, state)
%REFINE_NULLSPACE_CANDIDATE Correct and reorthogonalize a nullspace block.
%   The first size(state.U,1) rows are the coordinates associated with the
%   fixed square anchor. Each correction solves with the retained LU
%   factors so that Omega*candidate is driven back toward roundoff. Stop
%   as soon as the measured residual reaches the requested threshold.

    candidate = reorthogonalize(candidate, retained_basis);
    residual = relative_residual(Omega, candidate);
    for pass = 1:state.correction_passes
        defect = Omega * candidate;
        correction = state.U \ (state.L \ defect(state.P, :));
        anchor_size = size(state.U, 1);
        candidate(1:anchor_size, :) = ...
            candidate(1:anchor_size, :) - correction;
        candidate = reorthogonalize(candidate, retained_basis);
        residual = relative_residual(Omega, candidate);
        if residual <= state.residual_threshold
            break;
        end
    end
end

function candidate = reorthogonalize(candidate, retained_basis)
    if ~isempty(retained_basis)
        candidate = candidate - ...
            retained_basis * (retained_basis' * candidate);
    end
    [candidate, ~] = qr(candidate, 'econ');

    if ~isempty(retained_basis)
        candidate = candidate - ...
            retained_basis * (retained_basis' * candidate);
        [candidate, ~] = qr(candidate, 'econ');
    end
end

function residual = relative_residual(Omega, candidate)
    residual = norm(Omega * candidate, 'fro') / ...
        max(norm(Omega, 'fro'), realmin(class(Omega)));
end
