#!/bin/bash
#SBATCH -J hss-ls-unprecond
#SBATCH -o hss-ls-unprecond.%j.out
#SBATCH -e hss-ls-unprecond.%j.err
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 08:00:00

# The unpreconditioned GMRES count of the cavity Lippmann-Schwinger problem
# at N = 1024^2 (q = 32), the one entry of Table 7's n~_it column that
# needs more memory than a workstation. The count depends on N only
# (tolerance 1e-12, no restart, right-hand side f = A x_star as in
# Section 5.1). run_cavity_unpreconditioned.m computes the smaller sizes
# (23, 58, 211) locally; at this size a 16 GB machine holds only 715
# iterations of the unrestarted Krylov basis, which is not enough.
#
# No factorization is built. MATLAB allocates the N x maxit complex basis
# up front: 80 GB allows 4768 iterations, and the trend 23, 58, 211 puts
# the count near 1000. Each iteration is one 2D FFT matvec plus
# orthogonalization against the basis, so expect one to three hours. If
# the run still reports flag 1, the count is a lower bound; raise the
# budget (the node has 256 GB) and resubmit.
#
#   sbatch hpc/stampede3/lippmann_unprecond_1024_icx.slurm

set -euo pipefail
umask 027

code_dir="${HSS_CODE_DIR:-${SCRATCH}/Adaptive-Matfree-HSS}"
results_root="${HSS_RESULTS_ROOT:-${SCRATCH}/Adaptive-Matfree-HSS-results/diagnostics_refresh}"

if [[ ! -f "${code_dir}/run_cavity_unpreconditioned.m" ]]; then
    echo "Cannot find run_cavity_unpreconditioned.m under ${code_dir}" >&2
    exit 2
fi

mkdir -p "${results_root}/lippmann_unprecond_1024"
module reset
module load matlab

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${code_dir}"
echo "Started:  $(date --iso-8601=seconds)"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

matlab -batch "\
    try, \
        results = run_cavity_unpreconditioned(1024, 80e9); \
        save('${results_root}/lippmann_unprecond_1024/unpreconditioned_1024.mat', 'results'); \
        disp(results); \
    catch caught, \
        disp(getReport(caught, 'extended', 'hyperlinks', 'off')); exit(1); \
    end; exit(0);" 2>&1 | tee "${results_root}/lippmann_unprecond_1024/matlab.log"

echo "Finished: $(date --iso-8601=seconds)"
