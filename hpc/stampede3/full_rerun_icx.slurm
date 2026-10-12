#!/bin/bash
#SBATCH -J hss-full-rerun
#SBATCH -o hss-full-rerun.%A_%a.out
#SBATCH -e hss-full-rerun.%A_%a.err
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 12:00:00

# Runs every table of Section 5 (Tables 3-7, all 90 configurations) with
# one code version. The root of the HSS tree is sampled to at least 1.1
# times its dimension (ROOT_OVERSAMPLE in 2-D_Test_System/factorize_tree_2D.m;
# diagnostics/run_root_oversample_check.m measures the effect).
#
# Twelve tasks, the per-user running-job limit, balanced on wall time (the
# longest, 3D Poisson 512^2 at 1e-12, takes a few hours). Every call writes to its own directory: the driver
# keeps one configuration selection per output directory.
#
# The unpreconditioned Krylov counts depend on N alone and are already
# measured (Tables 5-7), so HSS_SKIP_UNPRECOND=1 skips the baseline; the
# tables keep their published n~_it columns. Table 5 is the Helmholtz strip
# family with its three matched streams and the X^rr diagnostic.
#
#   sbatch --array=0-11 hpc/stampede3/full_rerun_icx.slurm
#
# Copy back:
#   rsync -avz --progress <user>@stampede3.tacc.utexas.edu:'$SCRATCH/Adaptive-Matfree-HSS-results/full_rerun/' comparison_results/stampede3_full_rerun/

set -euo pipefail
umask 027

code_dir="${HSS_CODE_DIR:-${SCRATCH}/Adaptive-Matfree-HSS}"
results_root="${HSS_RESULTS_ROOT:-${SCRATCH}/Adaptive-Matfree-HSS-results/full_rerun}"
export HSS_SKIP_UNPRECOND=1
task="${SLURM_ARRAY_TASK_ID:-0}"

# Each call is "group|tolerances|n1d values|n values|subdirectory";
# an empty field means no restriction on that axis.
case "${task}" in
    0)  calls=("poisson_1d||||poisson_1d"
               "helmholtz_1d||||helmholtz_1d") ;;
    1)  calls=("poisson_2d||64 128||poisson_2d_64_128"
               "poisson_2d|1e-3 1e-6|256||poisson_2d_256_loose") ;;
    2)  calls=("poisson_2d|1e-9 1e-12|256||poisson_2d_256_tight") ;;
    3)  calls=("poisson_2d|1e-3 1e-6|512||poisson_2d_512_loose") ;;
    4)  calls=("poisson_2d|1e-9|512||poisson_2d_512_1e9") ;;
    5)  calls=("poisson_2d|1e-12|512||poisson_2d_512_1e12") ;;
    6)  calls=("matern_low||||matern_low") ;;
    7)  calls=("matern_high||||matern_high") ;;
    8)  calls=("lippmann|1e-3 1e-6 1e-9|128 256 512||lippmann_small"
               "lippmann|1e-12|128 256||lippmann_1e12_small") ;;
    9)  calls=("lippmann|1e-12|512||lippmann_512_1e12") ;;
    10) calls=("lippmann|1e-3 1e-6|1024||lippmann_1024_loose") ;;
    11) calls=("lippmann|1e-9|1024||lippmann_1024_1e9") ;;
    *)  echo "No task ${task}" >&2; exit 2 ;;
esac

if [[ ! -f "${code_dir}/run_paper_experiments.m" ]]; then
    echo "Cannot find run_paper_experiments.m under ${code_dir}" >&2
    exit 2
fi

module reset
module load matlab
module load gcc || echo "note: no gcc module loaded; mex may use the default compiler"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${code_dir}"
echo "Started:  $(date --iso-8601=seconds)"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Task:     ${task}"
module list

matlab -batch "if ~build_mex(), exit(3); end" || { echo "MEX build failed" >&2; exit 3; }

for call in "${calls[@]}"; do
    IFS='|' read -r group tols n1ds ns subdir <<<"${call}"
    results_dir="${results_root}/${subdir}"
    mkdir -p "${results_dir}"
    echo "--- ${group}  tol=[${tols}]  n1d=[${n1ds}]  n=[${ns}]  -> ${results_dir}"
    # Table 5 reports the X^rr conditioning of the Helmholtz strip (Section 5.3).
    if [[ "${group}" == "helmholtz_1d" ]]; then export HSS_XRR_DIAG=1; else unset HSS_XRR_DIAG; fi
    matlab -batch "\
        try, \
            run_paper_experiments('${results_dir}', 'legacy', '${group}', \
                [${tols}], [${n1ds}], 'householder', [${ns}]); \
        catch caught, \
            disp(getReport(caught, 'extended', 'hyperlinks', 'off')); exit(1); \
        end; exit(0);"
done

echo "Finished: $(date --iso-8601=seconds)"
