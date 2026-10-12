#!/bin/bash
#SBATCH -J hss-studies
#SBATCH -o hss-studies.%A_%a.out
#SBATCH -e hss-studies.%A_%a.err
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 24:00:00
#SBATCH --array=0-7%8

# The studies added alongside the main tables: the block-size sweep behind
# the block-size tables, the least-squares conditioning diagnostic, and the
# nullspace-method comparison. Eight independent tasks, each checkpointed, so
# a task that hits the wall clock resumes on resubmission.
#
# Queue limits (icx): 48 h max, 32 nodes per job, 12 jobs per user, 20 queued.
# This array is eight single-node tasks, so it fits alongside the main-table
# array without exhausting the job allowance.

set -euo pipefail
umask 027

# The 2D Poisson sweep is the slow one (two hours against 30-45 minutes for
# the others), so it runs as three tasks, one pruning scope
# each; merge_block_size_shards joins them afterwards (see
# hpc/stampede3/README.md).
tasks=(block_poisson_1d block_poisson_2d_all block_poisson_2d_leaf \
       block_poisson_2d_none block_matern block_lippmann \
       conditioning nullspace_comparison)
task="${tasks[${SLURM_ARRAY_TASK_ID}]}"

code_dir="${HSS_CODE_DIR:-${SCRATCH}/Adaptive-Matfree-HSS}"
results_root="${HSS_RESULTS_ROOT:-${SCRATCH}/Adaptive-Matfree-HSS-results/studies}"
results_dir="${results_root}/${task}"

if [[ ! -f "${code_dir}/run_block_size_study.m" ]]; then
    echo "Cannot find run_block_size_study.m under ${code_dir}" >&2
    exit 2
fi

mkdir -p "${results_dir}"
module reset
module load matlab
module load gcc || echo "note: no gcc module loaded; mex may use the default compiler"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${code_dir}"
echo "Started: $(date --iso-8601=seconds)"
echo "Task: ${task}   Host: $(hostname)   Results: ${results_dir}"
module list

matlab -batch "if ~build_mex(), exit(3); end" || { echo "MEX build failed" >&2; exit 3; }

case "${task}" in
    block_poisson_2d_*)
        scope="${task#block_poisson_2d_}"
        matlab -batch \
            "try, run_block_size_study('${results_dir}','poisson_2d',[],{'${scope}'}); \
             catch e, disp(getReport(e,'extended','hyperlinks','off')); exit(1); end; exit(0);"
        ;;
    block_*)
        group="${task#block_}"
        matlab -batch \
            "try, run_block_size_study('${results_dir}','${group}'); \
             catch e, disp(getReport(e,'extended','hyperlinks','off')); exit(1); end; exit(0);"
        ;;
    conditioning)
        # The manuscript's conditioning table: tolerance 1e-6, leaf 256,
        # b = 16 for the 1D problem and b = 32 for the 2D problems, as in
        # the Section 5 protocol. Prints the per-level tables into the .out
        # file and writes ls_conditioning_summary.csv (the table's columns:
        # equations, unknowns, rho, kappa measured, kappa_MP), the per-node
        # rows and the rank-search exits into ${results_dir}.
        matlab -batch \
            "try, run_ls_conditioning_diagnostic('pt_leaf',256,'tolerance',1e-6, \
                 'step_b',[16 32 32 32],'output_dir','${results_dir}'); \
             catch e, disp(getReport(e,'extended','hyperlinks','off')); exit(1); end; exit(0);"
        ;;
    nullspace_comparison)
        matlab -batch \
            "try, run_nullspace_comparison('${results_dir}',false,'high_accuracy'); \
             catch e, disp(getReport(e,'extended','hyperlinks','off')); exit(1); end; exit(0);"
        ;;
    *)
        echo "Unknown task ${task}" >&2; exit 2 ;;
esac

echo "Finished: $(date --iso-8601=seconds)"
