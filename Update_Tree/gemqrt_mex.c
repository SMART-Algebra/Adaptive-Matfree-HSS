/*
 * gemqrt_mex.c
 *
 * Apply the Q of a compact-WY QR (geqrt_mex) without forming it, via LAPACK
 * DGEMQRT (real) or ZGEMQRT (complex).
 *
 *   Z = gemqrt_mex(V, T, X)          % Z = Q * X
 *   Z = gemqrt_mex(V, T, X, 'T')     % Z = Q' * X  (conjugate transpose when
 *   Z = gemqrt_mex(V, T, X, 'C')     %              complex; 'T' and 'C' are
 *                                    %              the same request)
 *
 *   V : m-by-k Householder vectors from geqrt_mex (only the strictly lower
 *       part is read; the unit diagonal is implicit)
 *   T : NB-by-k block reflector factors from geqrt_mex (NB = size(T,1))
 *   X : m-by-q double
 *
 * V, T and X must be all real or all complex (a mix would silently drop
 * imaginary parts, so the caller casts up front).
 *
 * Q = H_1 ... H_k is m x m; this costs about 4*m*k*q flops instead of the
 * 4*(m^2*k - m*k^2 + k^3/3) of forming Q and then 2*m^2*q to apply it.
 *
 * Build:  build_mex   (mex -R2018a gemqrt_mex.c -lmwlapack)
 */

#include "mex.h"
#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
#define DGEMQRT dgemqrt
#define ZGEMQRT zgemqrt
#else
#define DGEMQRT dgemqrt_
#define ZGEMQRT zgemqrt_
#endif

extern void DGEMQRT(char *SIDE, char *TRANS, ptrdiff_t *M, ptrdiff_t *N,
                    ptrdiff_t *K, ptrdiff_t *NB,
                    double *V, ptrdiff_t *LDV, double *T, ptrdiff_t *LDT,
                    double *C, ptrdiff_t *LDC, double *WORK, ptrdiff_t *INFO);
extern void ZGEMQRT(char *SIDE, char *TRANS, ptrdiff_t *M, ptrdiff_t *N,
                    ptrdiff_t *K, ptrdiff_t *NB,
                    double *V, ptrdiff_t *LDV, double *T, ptrdiff_t *LDT,
                    double *C, ptrdiff_t *LDC, double *WORK, ptrdiff_t *INFO);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t m, k, nb, q, info = 0;
    char side = 'L', trans = 'N';
    double *Z, *work;
    int j, is_complex, any_complex, all_complex;
    size_t w;

    if (nrhs < 3 || nrhs > 4)
        mexErrMsgIdAndTxt("gemqrt_mex:nargin", "Inputs: V, T, X [, 'N', 'T' or 'C'].");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("gemqrt_mex:nargout", "One output.");
    for (j = 0; j < 3; j++)
        if (!mxIsDouble(prhs[j]) || mxIsSparse(prhs[j]))
            mexErrMsgIdAndTxt("gemqrt_mex:type", "V, T, X must be dense double.");
    any_complex = mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]) || mxIsComplex(prhs[2]);
    all_complex = mxIsComplex(prhs[0]) && mxIsComplex(prhs[1]) && mxIsComplex(prhs[2]);
    if (any_complex && !all_complex)
        mexErrMsgIdAndTxt("gemqrt_mex:mixed", "V, T, X must be all real or all complex.");
    is_complex = all_complex;
    w = is_complex ? 2 : 1;

    if (nrhs == 4) {
        char c[2];
        if (mxGetString(prhs[3], c, 2) != 0 || (c[0] != 'N' && c[0] != 'T' && c[0] != 'C'))
            mexErrMsgIdAndTxt("gemqrt_mex:trans", "Fourth input must be 'N', 'T' or 'C'.");
        trans = c[0];
        /* LAPACK: real GEMQRT takes 'T', complex takes 'C'. Both mean Q'. */
        if (trans != 'N') trans = is_complex ? 'C' : 'T';
    }

    m  = (ptrdiff_t) mxGetM(prhs[0]);
    k  = (ptrdiff_t) mxGetN(prhs[0]);
    nb = (ptrdiff_t) mxGetM(prhs[1]);
    q  = (ptrdiff_t) mxGetN(prhs[2]);
    if (k < 1 || m < k)
        mexErrMsgIdAndTxt("gemqrt_mex:shape", "V must be m-by-k with m >= k >= 1.");
    if ((ptrdiff_t) mxGetN(prhs[1]) != k || nb < 1 || nb > k)
        mexErrMsgIdAndTxt("gemqrt_mex:shape", "T must be NB-by-k with 1 <= NB <= k.");
    if ((ptrdiff_t) mxGetM(prhs[2]) != m)
        mexErrMsgIdAndTxt("gemqrt_mex:shape", "X must have as many rows as V.");

    plhs[0] = mxDuplicateArray(prhs[2]);
    if (q == 0) return;
    Z = (double *) mxGetData(plhs[0]);
    work = (double *) mxMalloc((size_t)(nb * q) * w * sizeof(double));

    /* V and T are read only; LAPACK's prototype is not const-qualified. */
    if (is_complex)
        ZGEMQRT(&side, &trans, &m, &q, &k, &nb,
                (double *) mxGetData(prhs[0]), &m, (double *) mxGetData(prhs[1]), &nb,
                Z, &m, work, &info);
    else
        DGEMQRT(&side, &trans, &m, &q, &k, &nb,
                (double *) mxGetData(prhs[0]), &m, (double *) mxGetData(prhs[1]), &nb,
                Z, &m, work, &info);
    mxFree(work);
    if (info != 0)
        mexErrMsgIdAndTxt("gemqrt_mex:lapack", "GEMQRT returned info = %d.", (int) info);
}
