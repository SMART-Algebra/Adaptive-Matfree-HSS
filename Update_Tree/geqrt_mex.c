/*
 * geqrt_mex.c
 *
 * Householder QR in compact WY form, via LAPACK DGEQRT (real) or ZGEQRT
 * (complex).
 *
 *   [V, T] = geqrt_mex(A)        % NB = n: one block, T is n x n
 *   [V, T] = geqrt_mex(A, NB)    % blocked T (NB x n), for gemqrt_mex
 *
 *   A : m-by-n double, real or complex, m >= n >= 1
 *   V : m-by-n, the Householder vectors, unit lower trapezoidal with the
 *       ones and the zeros above the diagonal stored explicitly
 *   T : NB-by-n, the upper triangular block reflector factors
 *
 * With NB = n, Q = H_1 ... H_n = I - V * T * V', where ' is the conjugate
 * transpose, so the WY pair used by apply_householder (Q = I - W * C') is
 * C = V, W = V * T. Same reflectors (LAPACK DLARFG/ZLARFG convention) as
 * MATLAB's qr and as the loop in compute_householder.
 *
 * Complex input uses the -R2018a interleaved layout, which matches Fortran
 * complex*16 element for element, so the same byte-level code serves both
 * precisions; only the LAPACK entry point differs.
 *
 * Build:  build_mex   (mex -R2018a geqrt_mex.c -lmwlapack)
 */

#include "mex.h"
#include <string.h>
#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
#define DGEQRT dgeqrt
#define ZGEQRT zgeqrt
#else
#define DGEQRT dgeqrt_
#define ZGEQRT zgeqrt_
#endif

extern void DGEQRT(ptrdiff_t *M, ptrdiff_t *N, ptrdiff_t *NB,
                   double *A, ptrdiff_t *LDA, double *T, ptrdiff_t *LDT,
                   double *WORK, ptrdiff_t *INFO);
extern void ZGEQRT(ptrdiff_t *M, ptrdiff_t *N, ptrdiff_t *NB,
                   double *A, ptrdiff_t *LDA, double *T, ptrdiff_t *LDT,
                   double *WORK, ptrdiff_t *INFO);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t m, n, nb, info = 0, i, j;
    double *V, *T, *work;
    int is_complex;
    size_t w;            /* doubles per element: 1 real, 2 complex */

    if (nrhs < 1 || nrhs > 2)
        mexErrMsgIdAndTxt("geqrt_mex:nargin", "Inputs: A [, NB].");
    if (nlhs > 2)
        mexErrMsgIdAndTxt("geqrt_mex:nargout", "Outputs: V, T.");
    if (!mxIsDouble(prhs[0]) || mxIsSparse(prhs[0]))
        mexErrMsgIdAndTxt("geqrt_mex:type", "A must be a dense double matrix.");

    is_complex = mxIsComplex(prhs[0]);
    w = is_complex ? 2 : 1;

    m = (ptrdiff_t) mxGetM(prhs[0]);
    n = (ptrdiff_t) mxGetN(prhs[0]);
    if (n < 1 || m < n)
        mexErrMsgIdAndTxt("geqrt_mex:shape", "Need m >= n >= 1.");

    nb = n;
    if (nrhs == 2) {
        nb = (ptrdiff_t) mxGetScalar(prhs[1]);
        if (nb < 1) nb = 1;
        if (nb > n) nb = n;
    }

    plhs[0] = mxDuplicateArray(prhs[0]);
    plhs[1] = mxCreateDoubleMatrix((mwSize) nb, (mwSize) n,
                                   is_complex ? mxCOMPLEX : mxREAL);
    V = (double *) mxGetData(plhs[0]);
    T = (double *) mxGetData(plhs[1]);
    work = (double *) mxMalloc((size_t)(nb * n) * w * sizeof(double));

    if (is_complex)
        ZGEQRT(&m, &n, &nb, V, &m, T, &nb, work, &info);
    else
        DGEQRT(&m, &n, &nb, V, &m, T, &nb, work, &info);
    mxFree(work);
    if (info != 0)
        mexErrMsgIdAndTxt("geqrt_mex:lapack", "GEQRT returned info = %d.", (int) info);

    /* R sits on and above the diagonal; keep only the reflectors, with their
       implicit unit diagonal made explicit. */
    for (j = 0; j < n; j++) {
        for (i = 0; i < j; i++) {
            V[(size_t)(i + j * m) * w] = 0.0;
            if (is_complex) V[(size_t)(i + j * m) * w + 1] = 0.0;
        }
        V[(size_t)(j + j * m) * w] = 1.0;
        if (is_complex) V[(size_t)(j + j * m) * w + 1] = 0.0;
    }
}
