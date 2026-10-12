function t = add_cols_time(cmd, val)
% addColsTime('reset')      -> zero the counter
% addColsTime('add', secs)  -> add elapsed seconds
% t = addColsTime('get')    -> read total seconds

persistent acc
if isempty(acc), acc = 0; end
switch cmd
    case 'reset', acc = 0;
    case 'add',   acc = acc + val;
    case 'get'    % do nothing, just return
    otherwise, error('Use ''reset'',''add'',''get''.');
end
t = acc;
end
