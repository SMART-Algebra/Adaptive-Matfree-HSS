function tests = test_checkpoint_provenance
%TEST_CHECKPOINT_PROVENANCE Checkpoints from other code are set aside, not resumed.
tests = functiontests(localfunctions);
end

function setupOnce(~)
here = fileparts(mfilename('fullpath'));
addpath(genpath(fileparts(here)));
end

function testCommitHashIsHexOrUnknown(testCase)
h = code_commit_hash();
verifyTrue(testCase, ischar(h) && ~isempty(h));
verifyTrue(testCase, strcmp(h, 'unknown') || all(isstrprop(h, 'xdigit')));
end

function testUnstampedCheckpointIsSetAside(testCase)
d = tempname; mkdir(d); cleanup = onCleanup(@() rmdir(d, 's')); %#ok<NASGU>
ckpt = fullfile(d, 'x_checkpoint.mat');
rows = struct('a', 1); save(ckpt, 'rows'); %#ok<NASGU>
companion = fullfile(d, 'x_raw.csv');
fid = fopen(companion, 'w'); fprintf(fid, 'a\n1\n'); fclose(fid);
[saved, resumed] = load_checkpoint_if_current(ckpt, {companion});
verifyFalse(testCase, resumed);
verifyEmpty(testCase, saved);
verifyFalse(testCase, isfile(ckpt));
verifyFalse(testCase, isfile(companion));
verifyTrue(testCase, isfile(fullfile(d, 'x_checkpoint.stale-unstamped.mat')));
verifyTrue(testCase, isfile(fullfile(d, 'x_raw.stale-unstamped.csv')));
end

function testCurrentCheckpointResumes(testCase)
d = tempname; mkdir(d); cleanup = onCleanup(@() rmdir(d, 's')); %#ok<NASGU>
ckpt = fullfile(d, 'x_checkpoint.mat');
rows = struct('a', 1); code_commit = code_commit_hash(); %#ok<NASGU>
save(ckpt, 'rows', 'code_commit');
[saved, resumed] = load_checkpoint_if_current(ckpt, {});
verifyTrue(testCase, resumed);
verifyEqual(testCase, saved.rows.a, 1);
verifyTrue(testCase, isfile(ckpt));
end

function testMismatchResumesOnlyWithOverride(testCase)
d = tempname; mkdir(d); cleanup = onCleanup(@() rmdir(d, 's')); %#ok<NASGU>
ckpt = fullfile(d, 'x_checkpoint.mat');
rows = struct('a', 1); code_commit = 'deadbee'; %#ok<NASGU>
save(ckpt, 'rows', 'code_commit');
old = getenv('HSS_RESUME_ACROSS_COMMITS');
restore = onCleanup(@() setenv('HSS_RESUME_ACROSS_COMMITS', old)); %#ok<NASGU>
setenv('HSS_RESUME_ACROSS_COMMITS', '1');
[~, resumed] = load_checkpoint_if_current(ckpt, {});
verifyTrue(testCase, resumed);
verifyTrue(testCase, isfile(ckpt));
setenv('HSS_RESUME_ACROSS_COMMITS', '');
[~, resumed] = load_checkpoint_if_current(ckpt, {});
verifyFalse(testCase, resumed);
verifyFalse(testCase, isfile(ckpt));
verifyTrue(testCase, isfile(fullfile(d, 'x_checkpoint.stale-deadbee.mat')));
end

function testComparisonDriverDoesNotResumeStaleCheckpoint(testCase)
d = tempname; mkdir(d); cleanup = onCleanup(@() rmdir(d, 's')); %#ok<NASGU>
rows = struct('config_id', 1, 'seed', 0, 'method', "naive"); %#ok<NASGU>
save(fullfile(d, 'nullspace_comparison_checkpoint.mat'), 'rows');
out = evalc('run_nullspace_comparison(d, true, ''high_accuracy'');');
verifyTrue(testCase, ~contains(out, 'Skipping completed run'));
verifyTrue(testCase, isfile(fullfile(d, 'nullspace_comparison_checkpoint.stale-unstamped.mat')));
raw = read_results_table(fullfile(d, 'nullspace_comparison_raw.csv'));
verifyTrue(testCase, all(raw.code_commit == string(code_commit_hash())));
end
