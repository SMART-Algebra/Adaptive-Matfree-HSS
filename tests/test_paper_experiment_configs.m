function tests = test_paper_experiment_configs
%TEST_PAPER_EXPERIMENT_CONFIGS Verify every suite can build before execution.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase) %#ok<INUSD>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(here));
end

function testConfigurationValidation(testCase) %#ok<INUSD>
    modes = {'smoke', 'pilot', 'laptop', 'legacy', 'scaling'};
    for mode_id = 1:numel(modes)
        output_dir = tempname;
        cleanup = onCleanup(@() remove_if_present(output_dir));
        try
            run_paper_experiments(output_dir, modes{mode_id}, ...
                'invalid_test_group');
        catch caught
            assert(contains(caught.message, 'Unknown group(s)'), ...
                'Configuration construction failed before group validation.');
            clear cleanup;
            continue;
        end
        error('Expected invalid-group validation for mode %s.', modes{mode_id});
    end

    output_dir = tempname;
    cleanup = onCleanup(@() remove_if_present(output_dir));
    try
        run_paper_experiments(output_dir, 'legacy', ...
            'matern_low', 1e-7);
    catch caught
        assert(contains(caught.message, ...
            'Requested tolerance(s) are unavailable'), ...
            'Tolerance filtering did not reject an unavailable value.');
        fprintf('Tolerance-filter validation passed.\n');
        clear cleanup;
    end
    if isfolder(output_dir)
        error('Expected validation to reject an unavailable tolerance.');
    end

    output_dir = tempname;
    cleanup = onCleanup(@() remove_if_present(output_dir));
    try
        run_paper_experiments(output_dir, 'legacy', ...
            'poisson_2d', 1e-6, 96);
    catch caught
        assert(contains(caught.message, ...
            'Requested grid width(s) are unavailable'), ...
            'Grid-width filtering did not reject an unavailable value.');
        clear cleanup;
        fprintf('Grid-width-filter validation passed.\n');
        fprintf('Paper experiment configuration tests passed.\n');
        return;
    end
    error('Expected validation to reject an unavailable grid width.');

end

function remove_if_present(path)
    if isfolder(path)
        rmdir(path, 's');
    end
end
