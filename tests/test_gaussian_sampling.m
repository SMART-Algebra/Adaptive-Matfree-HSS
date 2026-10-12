function tests = test_gaussian_sampling
%TEST_GAUSSIAN_SAMPLING Tests for real and circular complex sketches.
    tests = functiontests(localfunctions);
end

function testRealSamples(testCase)
    rng(11, 'twister');
    Omega = draw_gaussian(40, 30, 'real');
    verifySize(testCase, Omega, [40, 30]);
    verifyTrue(testCase, isreal(Omega));
end

function testCircularComplexSamples(testCase)
    rng(11, 'twister');
    Omega = draw_gaussian(400, 300, 'complex');
    verifySize(testCase, Omega, [400, 300]);
    verifyFalse(testCase, isreal(Omega));
    verifyLessThan(testCase, abs(mean(Omega(:))), 2e-2);
    verifyLessThan(testCase, abs(mean(abs(Omega(:)).^2) - 1), 2e-2);
    verifyLessThan(testCase, abs(var(real(Omega(:)), 1) - 0.5), 2e-2);
    verifyLessThan(testCase, abs(var(imag(Omega(:)), 1) - 0.5), 2e-2);
end

function testComplexSamplesReachMatvec(testCase)
    add_cols_num('reset');
    add_cols_time('reset');
    [Omega, Y] = add_columns_at_leaves(7, @(x) x, 12, 'complex');
    verifySize(testCase, Omega, [12, 7]);
    verifyFalse(testCase, isreal(Omega));
    verifyEqual(testCase, Y, Omega, 'AbsTol', 10 * eps);
    verifyEqual(testCase, add_cols_num('get'), 7);
    add_cols_num('reset');
    add_cols_time('reset');
end

function testInvalidDistribution(testCase)
    verifyError(testCase, @() draw_gaussian(2, 2, 'unsupported'), ...
        'draw_gaussian:InvalidDistribution');
end
