# Running the paper's experiments on Stampede3

The three job scripts here reproduce Section 5 and the Supplementary
Material of the paper on TACC's Stampede3. Each task runs on one Ice Lake
node (`icx` queue, 80 cores, 256 GB). The two array jobs build the MEX
kernels first and abort if the build fails; the unpreconditioned-count job
needs no kernel.

## Staging

```bash
cd $SCRATCH
git clone https://github.com/SMART-Algebra/Adaptive-Matfree-HSS.git
```

The scripts read the code from `$SCRATCH/Adaptive-Matfree-HSS` and write
results below `$SCRATCH/Adaptive-Matfree-HSS-results/`; set `HSS_CODE_DIR`
and `HSS_RESULTS_ROOT` to change either. Pass your allocation and, if you
want notifications, your email on the `sbatch` line.

## Jobs

```bash
# Tables 3-7: all 90 configurations, twelve tasks, about half a day
sbatch -A <allocation> --array=0-11 hpc/stampede3/full_rerun_icx.slurm

# Supplement: block-size study (tasks 0-5), least-squares conditioning (6),
# nullspace comparison (7)
sbatch -A <allocation> --array=0-7 hpc/stampede3/studies_icx.slurm

# Unpreconditioned GMRES count of the cavity problem at N = 1024^2
sbatch -A <allocation> hpc/stampede3/lippmann_unprecond_1024_icx.slurm
```

Add `--mail-user=<email> --mail-type=END,FAIL` to be notified. The
unpreconditioned Krylov counts of the other rows are computed by the local
scripts named in the top-level README; `full_rerun_icx.slurm` sets
`HSS_SKIP_UNPRECOND=1` so that the table runs do not repeat them.

## Copying results back

```bash
rsync -avz <user>@stampede3.tacc.utexas.edu:'$SCRATCH/Adaptive-Matfree-HSS-results/full_rerun/' comparison_results/stampede3_full_rerun/
```

Then, in MATLAB, `collect_paper_experiment_results('comparison_results/stampede3_full_rerun')`
merges the per-task CSV files into one table. The studies job writes below
`$SCRATCH/Adaptive-Matfree-HSS-results/studies/`; copy it the same way. Its
2D Poisson block-size sweep runs as three tasks, one per pruning scope, and
`merge_block_size_shards` (see its help text) joins the three directories
into one before `write_block_size_summary` is applied. `comparison_results/`
is ignored by git.
