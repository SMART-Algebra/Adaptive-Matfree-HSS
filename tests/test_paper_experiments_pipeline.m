function tests = test_paper_experiments_pipeline
%TEST_PAPER_EXPERIMENTS_PIPELINE Run the table driver end to end.
%   The Stampede3 smoke job of 2026-09-12 found that every run of
%   run_paper_experiments failed after its factorization had finished,
%   because the row builders read a variable that was not in their scope,
%   and that the failure path crashed the same way, so no log was written.
%   These tests run the driver on its tiny 'smoke' configuration through
%   both paths.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase) %#ok<INUSD>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(here));
end

function testSmokeSuiteWritesOkRows(testCase)
    out = tempname;
    cleanup = onCleanup(@() remove_if_present(out)); %#ok<NASGU>
    [~, raw] = evalc('run_paper_experiments(out, ''smoke'', [], [], [], ''householder'')');
    verifyEqual(testCase, height(raw), 1);
    verifyEqual(testCase, string(raw.status(1)), "ok");
    verifyEqual(testCase, string(raw.method(1)), "householder");
    verifyTrue(testCase, isfile(fullfile(out, 'paper_results_raw.csv')));
    verifyGreaterThan(testCase, raw.n_matvec(1), 0);
    verifyLessThan(testCase, raw.solve_residual(1), 1e-4);
    verifyGreaterThan(testCase, raw.factor_storage_mb(1), 0);
end

function testFailureIsRecordedNotFatal(testCase)
    % An unknown method makes main() throw inside the driver's try block.
    % The driver must record that as a failed row, write the CSV, and only
    % then raise its own summary error.
    out = tempname;
    cleanup = onCleanup(@() remove_if_present(out)); %#ok<NASGU>
    caught = [];
    try
        evalc('run_paper_experiments(out, ''smoke'', [], [], [], ''no_such_method'')');
    catch caught
    end
    verifyNotEmpty(testCase, caught, 'The driver should report the failed configuration.');
    verifyTrue(testCase, contains(caught.message, 'configuration(s) failed'));
    csv = fullfile(out, 'paper_results_raw.csv');
    verifyTrue(testCase, isfile(csv), 'The raw CSV was not written on failure.');
    raw = read_results_table(csv);
    verifyEqual(testCase, height(raw), 1);
    verifyEqual(testCase, raw.status(1), "failed");
    verifyTrue(testCase, contains(raw.error_message(1), 'Unknown nullspace method'));
end

function remove_if_present(path)
    if isfolder(path)
        rmdir(path, 's');
    end
end
