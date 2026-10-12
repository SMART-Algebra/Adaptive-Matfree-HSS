function U_struct = wy_or_identity(U_b)
%WY_OR_IDENTITY WY pair for a node basis, or the identity when it is square.
%   A square basis spans the whole node: the rank search reached the node
%   dimension, every index is a skeleton index, and the block is passed to
%   the parent uncompressed. Rotating it by an arbitrary orthogonal factor
%   is then pure cost, so this returns empty W and C instead.
%
%   That is not a special case downstream. apply_householder forms
%   Q = I - W*C', and with W and C of size m-by-0 the product W*C' is the
%   m-by-m zero matrix, so Q is exactly the identity and every call site
%   becomes a no-op with no change of its own. The node also stores nothing
%   for U, which is worth 2*m^2 + 2*m numbers per saturated node.
%
%   There is a second reason to prefer the identity here. The alternative
%   basis comes from a QR of the samples, so it depends on Omega; applying
%   it would hand the parent a test block that is no longer Gaussian. The
%   identity leaves the parent's test matrix exactly as drawn, which is the
%   hypothesis the stopping test of Section 4 relies on.

    if size(U_b, 1) == size(U_b, 2)
        empty_basis = zeros(size(U_b, 1), 0, 'like', U_b);
        U_struct = struct('W', empty_basis, 'C', empty_basis);
        return;
    end

    [W, C] = compute_householder(U_b);
    U_struct = struct('W', W, 'C', C);
end
