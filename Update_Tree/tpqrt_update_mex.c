/*
 * tpqrt_update_mex.c
 *
 * One blocked Householder row-update of a QR factorization, via the LAPACK
 * kernels DTPQRT ("triangle on top of square") and DTPMQRT.
 *
 *   [R_new, Q1_new, DeltaQ] = tpqrt_update_mex(R, B, Q1)
 *   [R_new, Q1_new, DeltaQ] = tpqrt_update_mex(R, B, Q1, NB)   % blocking factor
 *
 *   R  : m-by-m upper triangular, the R factor of Omega_old'
 *   B  : b-by-m, the new rows of Omega' (i.e. DeltaOmega')
 *   Q1 : p-by-m, the range part of the QR factor of Omega_old'
 *
 *   R_new  : m-by-m upper triangular, R factor of Omega_new'
 *   Q1_new : (p+b)-by-m, updated range part
 *   DeltaQ : (p+b)-by-b, the new orthonormal nullspace directions
 *
 * The reflectors touch only pivot row j and the b appended rows, so the
 * previously computed nullspace columns are preserved exactly when padded
 * with b zero rows -- they are never passed to LAPACK at all.
 *
 * Real and complex are both supported: DTPQRT/DTPMQRT for real input,
 * ZTPQRT/ZTPMQRT for complex. For complex Omega the relevant factorization
 * is that of Omega^H, so the caller must pass the conjugate transpose.
 *
 * Build:  mex -R2018a tpqrt_update_mex.c -lmwlapack
 */

#include "mex.h"
#include <string.h>
#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
#define DTPQRT  dtpqrt
#define DTPMQRT dtpmqrt
#define ZTPQRT  ztpqrt
#define ZTPMQRT ztpmqrt
#else
#define DTPQRT  dtpqrt_
#define DTPMQRT dtpmqrt_
#define ZTPQRT  ztpqrt_
#define ZTPMQRT ztpmqrt_
#endif

extern void DTPQRT(ptrdiff_t *M, ptrdiff_t *N, ptrdiff_t *L, ptrdiff_t *NB,
                   double *A, ptrdiff_t *LDA, double *B, ptrdiff_t *LDB,
                   double *T, ptrdiff_t *LDT, double *WORK, ptrdiff_t *INFO);

extern void DTPMQRT(char *SIDE, char *TRANS, ptrdiff_t *M, ptrdiff_t *N,
                    ptrdiff_t *K, ptrdiff_t *L, ptrdiff_t *NB,
                    double *V, ptrdiff_t *LDV, double *T, ptrdiff_t *LDT,
                    double *A, ptrdiff_t *LDA, double *B, ptrdiff_t *LDB,
                    double *WORK, ptrdiff_t *INFO);

extern void ZTPQRT(ptrdiff_t *M, ptrdiff_t *N, ptrdiff_t *L, ptrdiff_t *NB,
                   double *A, ptrdiff_t *LDA, double *B, ptrdiff_t *LDB,
                   double *T, ptrdiff_t *LDT, double *WORK, ptrdiff_t *INFO);

extern void ZTPMQRT(char *SIDE, char *TRANS, ptrdiff_t *M, ptrdiff_t *N,
                    ptrdiff_t *K, ptrdiff_t *L, ptrdiff_t *NB,
                    double *V, ptrdiff_t *LDV, double *T, ptrdiff_t *LDT,
                    double *A, ptrdiff_t *LDA, double *B, ptrdiff_t *LDB,
                    double *WORK, ptrdiff_t *INFO);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t m, b, p, pb, nb, zero = 0, info = 0, ldv, ldc;
    double *R, *Bv, *Q1new, *DeltaQ, *T, *work;
    const double *Rin, *Bin, *Q1in;
    char side = 'R', trans = 'N';
    ptrdiff_t j;
    int is_complex;
    size_t esz;          /* bytes per matrix element: 8 real, 16 complex */
    mxComplexity cplx;

    if (nrhs < 3 || nrhs > 4)
        mexErrMsgIdAndTxt("tpqrt_update:nargin", "Inputs: R, B, Q1 [, NB].");
    if (nlhs > 3)
        mexErrMsgIdAndTxt("tpqrt_update:nargout", "At most three outputs.");

    for (j = 0; j < 3; j++) {
        if (!mxIsDouble(prhs[j]) || mxIsSparse(prhs[j]))
            mexErrMsgIdAndTxt("tpqrt_update:type",
                              "Inputs must be dense double matrices.");
    }
    /* All three must agree: mixing real and complex would silently drop the
       imaginary parts, so require the caller to cast up front. */
    is_complex = mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]) || mxIsComplex(prhs[2]);
    if (is_complex && !(mxIsComplex(prhs[0]) && mxIsComplex(prhs[1]) && mxIsComplex(prhs[2])))
        mexErrMsgIdAndTxt("tpqrt_update:mixed",
                          "R, B and Q1 must be all real or all complex.");
    esz  = is_complex ? 2 * sizeof(double) : sizeof(double);
    cplx = is_complex ? mxCOMPLEX : mxREAL;

    m = (ptrdiff_t) mxGetM(prhs[0]);
    if ((ptrdiff_t) mxGetN(prhs[0]) != m)
        mexErrMsgIdAndTxt("tpqrt_update:shape", "R must be square.");
    b = (ptrdiff_t) mxGetM(prhs[1]);
    if ((ptrdiff_t) mxGetN(prhs[1]) != m)
        mexErrMsgIdAndTxt("tpqrt_update:shape", "B must be b-by-m.");
    p = (ptrdiff_t) mxGetM(prhs[2]);
    if ((ptrdiff_t) mxGetN(prhs[2]) != m)
        mexErrMsgIdAndTxt("tpqrt_update:shape", "Q1 must be p-by-m.");
    if (m < 1 || b < 1)
        mexErrMsgIdAndTxt("tpqrt_update:shape", "Need m >= 1 and b >= 1.");

    pb = p + b;
    if (nrhs == 4) {
        nb = (ptrdiff_t) mxGetScalar(prhs[3]);
        if (nb < 1) nb = 1;
        if (nb > m) nb = m;
    } else {
        /* Narrow block reflectors beat wide ones here: the update is skinny,
           so a smaller nb keeps more of the work in the level-3 kernel. */
        nb = b / 2; if (nb < 4) nb = 4; if (nb > m) nb = m;
    }

    /* The -R2018a interleaved layout matches Fortran complex*16 exactly, so
       the same byte-level code serves both precisions. */
    Rin  = (const double *) mxGetData(prhs[0]);
    Bin  = (const double *) mxGetData(prhs[1]);
    Q1in = (const double *) mxGetData(prhs[2]);

    /* Outputs: R_new (m x m), Q1_new ((p+b) x m), DeltaQ ((p+b) x b). */
    plhs[0] = mxCreateDoubleMatrix((mwSize) m,  (mwSize) m, cplx);
    plhs[1] = mxCreateDoubleMatrix((mwSize) pb, (mwSize) m, cplx);
    plhs[2] = mxCreateDoubleMatrix((mwSize) pb, (mwSize) b, cplx);
    R      = (double *) mxGetData(plhs[0]);
    Q1new  = (double *) mxGetData(plhs[1]);
    DeltaQ = (double *) mxGetData(plhs[2]);

    memcpy(R, Rin, (size_t)(m * m) * esz);

    /* Q1new <- [Q1; 0_{b x m}]   (column major, so copy column by column). */
    for (j = 0; j < m; j++)
        memcpy((char *) Q1new + (size_t)(j * pb) * esz,
               (const char *) Q1in + (size_t)(j * p) * esz, (size_t) p * esz);

    /* DeltaQ <- [0_{p x b}; I_b]; for complex only the real part is set. */
    for (j = 0; j < b; j++)
        DeltaQ[((size_t)(j * pb + (p + j)) * esz) / sizeof(double)] = 1.0;

    /* B is overwritten with the reflector vectors V, so take a copy. */
    Bv = (double *) mxMalloc((size_t)(b * m) * esz);
    memcpy(Bv, Bin, (size_t)(b * m) * esz);

    T    = (double *) mxCalloc((size_t)(nb * m), esz);
    work = (double *) mxMalloc((size_t)(nb * (pb > m ? pb : m)) * esz);

    /* [R; B] = Q [R_new; 0], with L = 0 (square lower block). */
    if (is_complex)
        ZTPQRT(&b, &m, &zero, &nb, R, &m, Bv, &b, T, &nb, work, &info);
    else
        DTPQRT(&b, &m, &zero, &nb, R, &m, Bv, &b, T, &nb, work, &info);
    if (info != 0) {
        mxFree(Bv); mxFree(T); mxFree(work);
        mexErrMsgIdAndTxt("tpqrt_update:dtpqrt", "TPQRT failed, info = %d.", (int) info);
    }

    /* [Q1new DeltaQ] <- [Q1new DeltaQ] * Q. */
    ldv = b;
    ldc = pb;
    if (is_complex)
        ZTPMQRT(&side, &trans, &pb, &b, &m, &zero, &nb,
                Bv, &ldv, T, &nb, Q1new, &ldc, DeltaQ, &ldc, work, &info);
    else
        DTPMQRT(&side, &trans, &pb, &b, &m, &zero, &nb,
                Bv, &ldv, T, &nb, Q1new, &ldc, DeltaQ, &ldc, work, &info);
    if (info != 0) {
        mxFree(Bv); mxFree(T); mxFree(work);
        mexErrMsgIdAndTxt("tpqrt_update:dtpmqrt", "TPMQRT failed, info = %d.", (int) info);
    }

    /* Zero the strict lower triangle of R_new, which still holds reflector data. */
    { ptrdiff_t i; size_t w = esz / sizeof(double);
      for (j = 0; j < m; j++)
          for (i = j + 1; i < m; i++) {
              R[(size_t)(j * m + i) * w] = 0.0;
              if (is_complex) R[(size_t)(j * m + i) * w + 1] = 0.0;
          } }

    mxFree(Bv); mxFree(T); mxFree(work);
}
