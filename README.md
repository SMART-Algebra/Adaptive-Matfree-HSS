# Adaptive-Matfree-HSS

MATLAB implementation of the algorithm in

> Elly Do and Chao Chen, *Simultaneous Matrix-Free HSS Compression and
> Factorization with Adaptive Ranks*, submitted to the Journal of Scientific
> Computing, 2026.

The code compresses a dense matrix with hierarchically low-rank off-diagonal
structure into Hierarchically Semi-Separable (HSS) form and factorizes it at
the same time, in one upward pass over the HSS tree. It accesses the matrix
only through products with the matrix and, when needed, its transpose or
adjoint. The off-diagonal ranks are chosen during the factorization from a
prescribed tolerance. The drivers and job scripts in this repository
regenerate every table of the paper and the data behind its figures.

## Versions

Release [`v1.0.0`](https://github.com/SMART-Algebra/Adaptive-Matfree-HSS/releases/tag/v1.0.0)
is the code that produced the results in the paper. Later commits on `main`
may change defaults, timings, or results; check out `v1.0.0` to reproduce
the paper:

```bash
git clone --branch v1.0.0 https://github.com/SMART-Algebra/Adaptive-Matfree-HSS.git
```

## Citation

```bibtex
@article{do2026adaptive,
  author = {Do, Elly and Chen, Chao},
  title  = {Simultaneous Matrix-Free {HSS} Compression and Factorization with Adaptive Ranks},
  year   = {2026},
  note   = {Submitted to the Journal of Scientific Computing}
}
```

## Requirements

- MATLAB R2022a or later, no toolboxes. The paper's runs used R2025b on
  Stampede3; the code is also tested with R2024a on macOS.
- A C compiler for the MEX kernels (`mex -setup C`). The kernels call
  LAPACK through MATLAB's own library, so nothing else is needed.

```matlab
addpath(genpath(pwd));
build_mex
```

`build_mex` compiles three LAPACK kernels: the Householder row update that
extends a nullspace basis (`tpqrt_update_mex`), the compact-WY QR of a node
basis (`geqrt_mex`), and its application (`gemqrt_mex`). Each has a MATLAB
fallback, so the code runs without them, but the fallbacks are much slower
and the code warns once when one is used. Build the kernels before any
timed run.

## Quick start

`main(params)` builds a test matrix, computes the adaptive HSS factorization,
solves one linear system with it, and runs the preconditioned Krylov solve.
Every field has a default; the example below factorizes the Schur
complement of the 2D Poisson equation on a separator of $2^6 \cdot 256$
points at tolerance $10^{-6}$.

```matlab
params = struct( ...
    'dim',            1, ...              % 1: a 1D index set, 2: a 2D index set
    'kernel_type',    'poisson_schur', ...
    'user_tolerance', 1e-6, ...
    'step_b',         16, ...             % sampling block size b
    'num_levels',     7, ...              % leaves at level 7; N = 2^6 * pt_leaf
    'pt_leaf',        256, ...            % leaf size m
    'method',         'householder');     % the nullspace update of the paper
[node, stats] = main(params);
fprintf('matvecs %d, relres %.2e, factor storage %.0f MB\n', ...
    stats.n_matvec, stats.solve_residual, stats.factor_storage_mb);
```

Kernels: `poisson_schur` and `helmholtz_schur` (Schur complements of the 2D
and 3D Poisson and Helmholtz equations, `dim` selects the separator
dimension), `matern` with `kernel_params = struct('nu', 1.5, 'ell', 0.1)`,
and `helmholtz` with `kernel_params = struct('potential', 'cavity')` for the
complex-symmetric Lippmann--Schwinger equation (`dim = 2`). Leave
`kernel_type` empty to pick one from a menu. In 2D, the tree has four
children per node and $N = (2^{L-1} \sqrt{m})^2$ for `num_levels` $L$.
In 2D, `pt_leaf` must be a perfect square, since a leaf is a
$\sqrt{m}\times\sqrt{m}$ patch. `build_test_system.m` documents the remaining
kernel parameters.

## Reproducing the paper

Every table comes from `run_paper_experiments.m`, which runs a named family
of configurations with checkpoints and writes `paper_results_raw.csv` in the
output directory. The `legacy` mode holds the configurations of the paper.

```matlab
run_paper_experiments('results/poisson_1d', 'legacy', 'poisson_1d');  % Table 3
```

| Paper | Family or script |
|---|---|
| Table 3, and the data of Figure 6 | `poisson_1d` |
| Table 4 | `poisson_2d` |
| Table 5 | `helmholtz_1d` (three seeds per configuration; the table reports medians) |
| Table 6 | `matern_low`, `matern_high` |
| Table 7 | `lippmann` |
| Unpreconditioned counts $\tilde n_{\mathrm{it}}$ | `run_helmholtz_strip_unpreconditioned.m`, `run_matern_unpreconditioned.m`, `run_cavity_unpreconditioned.m` |
| Figure 7 | `plot_cavity_fields(256, out_dir)` and `plot_cavity_fields(512, out_dir)` |
| Tables S2 and S3 of the supplement (block size, pruning scope) | `run_block_size_study.m` |
| Table 8 (Table S1 of the supplement) | `run_ls_conditioning_diagnostic.m` |

`collect_paper_experiment_results(results_root)` merges the family CSVs
under one root into one table with every metric the paper reports.

The largest configurations need a node with 256 GB of memory. The job
scripts under `hpc/stampede3/` run the whole of Section 5 as one job array
on TACC's Stampede3 and are documented in `hpc/stampede3/README.md`. The
`pilot` mode of `run_paper_experiments` runs one small configuration for
each of `poisson_1d`, `poisson_2d`, `matern_low` and `lippmann` and fits
on a laptop.

## Main entry points

```
main.m                    one factorization and solve from a parameter struct
build_test_system.m       the test matrices and their fast matvecs
build_mex.m               compiles the LAPACK kernels
run_paper_experiments.m   the paper's configurations, with checkpoints
run_*.m                   the other drivers named above; the remaining .m
                          files at the root are their helpers
Update_Tree/              adaptive basis construction: sampling, stopping test,
                          nullspace row update, pruning (MEX sources here)
Build_Tree/               compression of one node and elimination of its
                          redundant variables
Solve_Matvec/             matvecs, the factorized solve, and the Krylov solvers
1-D_Test_System/          1D index sets: tree, Schur complements, DST solver
2-D_Test_System/          2D index sets: quadtree and 2D Schur complements
diagnostics/              studies behind design choices in the code (see below)
hpc/stampede3/            Slurm job scripts for the paper's runs
tests/                    unit tests
```

## Tests

```matlab
addpath(genpath(pwd)); build_mex;
runtests('tests')
```

The tests cover the nullspace row update (real and complex, MEX and
fallback), the LAPACK kernels, the stopping and freeze rules, rank pruning,
factor storage, and the experiment driver's checkpoints.

## Diagnostics

The scripts in `diagnostics/` reproduce the measurements that the code
comments cite for four design choices: the level-wide stopping floor
(`run_stop_floor_diagnostic.m`), sampling the root beyond its dimension
(`run_root_oversample_check.m`), and the transverse width of the Schur
complement strips (`run_strip_width_study.m`, `run_poisson_width_study.m`). At the repository
root, `run_nullspace_comparison.m` and `run_rank_pruning_diagnostic.m`
compare the nullspace update with the alternatives discussed in the paper.

## License

MIT. See [LICENSE](LICENSE).
