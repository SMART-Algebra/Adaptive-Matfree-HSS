function tests = test_level_rank_budget
%TEST_LEVEL_RANK_BUDGET A marginal node must not force the level to full rank.
%   Regression for the 2026-09-13 finding. The rank search stops either
%   because every node passed, because a node exhausted its rank budget, or
%   because the level ran out of samples while a node was marginal. Only the
%   middle case may store the level at the node dimension.
tests = functiontests(localfunctions);
end

function setupOnce(~)
here = fileparts(mfilename('fullpath'));
addpath(genpath(fileparts(here)));
end

function testNoFailuresIsNotExhaustion(testCase)
verifyFalse(testCase, level_exhausted_rank_budget(false(4, 1), [240; 240; 16; 16], 256, 16));
end

function testMarginalNarrowNodeIsNotExhaustion(testCase)
% The 1D Poisson N = 262144 case: one node of many fails with a 16-column
% basis while the certifiable ceiling is 240.
fail = false(1024, 1); fail(501) = true;
width = repmat(16, 1024, 1); width(501) = 32;
verifyFalse(testCase, level_exhausted_rank_budget(fail, width, 256, 16));
end

function testNodeAtTheCeilingIsExhaustion(testCase)
fail = [false; true; false];
width = [16; 240; 16];
verifyTrue(testCase, level_exhausted_rank_budget(fail, width, 256, 16));
verifyTrue(testCase, level_exhausted_rank_budget(fail, [16; 256; 16], 256, 16));
end

function testFailingNodeBelowCeilingIsNotExhaustion(testCase)
verifyTrue(testCase, level_exhausted_rank_budget([true; false], [224; 16], 256, 32));
verifyFalse(testCase, level_exhausted_rank_budget([true; false], [208; 16], 256, 32));
end

function testCeilingUsesTheRequestedStep(testCase)
% A wider step lowers the ceiling, so the same node can be exhausted at
% b = 32 and merely marginal at b = 16.
verifyTrue(testCase, level_exhausted_rank_budget(true, 224, 256, 32));
verifyFalse(testCase, level_exhausted_rank_budget(true, 224, 256, 16));
end
