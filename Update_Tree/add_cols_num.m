function c = add_cols_num(cmd, val)
% add_cols_num('reset')     -> zero the matvec counter
% add_cols_num('add', n)    -> add n matvecs (e.g., step_b)
% c = add_cols_num('get')   -> read total number of matvecs

persistent acc
if isempty(acc), acc = 0; end
switch cmd
    case 'reset'
        acc = 0;
    case 'add'
        acc = acc + val;
    case 'get'
        % do nothing, just return
    otherwise
        error('Use ''reset'', ''add'', or ''get''.');
end
c = acc;
end
