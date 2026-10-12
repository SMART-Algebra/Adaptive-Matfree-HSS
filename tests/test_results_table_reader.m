function tests = test_results_table_reader
%TEST_RESULTS_TABLE_READER Provenance columns must survive the CSV round trip.
%   Both cases below were observed, not imagined: a block_size_raw.csv was
%   read as three columns on R2024a because level_ranks is packed with '|',
%   and a short commit hash made only of digits (such as 4067627), which
%   readtable infers as numeric and
%   which then errors when compared against a string.
tests = functiontests(localfunctions);
end

function testAllDigitCommitStaysText(testCase)
d = tempname; mkdir(d); cleanup = onCleanup(@() rmdir(d, 's')); %#ok<NASGU>
f = fullfile(d, 'rows.csv');
T = table(["4067627"; "4067627"], [1; 2], 'VariableNames', {'code_commit', 'n'});
writetable(T, f);
R = read_results_table(f);
verifyTrue(testCase, isstring(R.code_commit));
verifyTrue(testCase, all(R.code_commit == "4067627"));
end

function testLeadingZeroCommitIsNotTruncated(testCase)
d = tempname; mkdir(d); cleanup = onCleanup(@() rmdir(d, 's')); %#ok<NASGU>
f = fullfile(d, 'rows.csv');
writetable(table(["0012345"], [1], 'VariableNames', {'code_commit', 'n'}), f);
R = read_results_table(f);
verifyEqual(testCase, R.code_commit(1), "0012345");
end

function testPipePackedColumnDoesNotBreakDelimiter(testCase)
d = tempname; mkdir(d); cleanup = onCleanup(@() rmdir(d, 's')); %#ok<NASGU>
f = fullfile(d, 'rows.csv');
writetable(table(["poisson_1d"], ["7:256:18|6:36:18|5:36:18"], [32], ...
    'VariableNames', {'case_name', 'level_ranks', 'step_b'}), f);
R = read_results_table(f);
verifyEqual(testCase, width(R), 3);
verifyEqual(testCase, R.level_ranks(1), "7:256:18|6:36:18|5:36:18");
verifyEqual(testCase, R.step_b(1), 32);
end
