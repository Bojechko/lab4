#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include "lab4.cuh"

#define USE_SIMPLE_FILTER 0

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*
    Transpose a 2D array (see SDK transpose example)
*/
extern "C"
void transpose(uint *d_src, uint *d_dest, uint width, int height)
{
    dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    d_transpose<<< grid, threads >>>(d_dest, d_src, width, height);
    getLastCudaError("Kernel execution failed");
}

// 8-bit RGBA version
extern "C"
void gaussianFilterRGBA(uint *d_src, uint *d_dest, uint *d_temp, int width, int height, float sigma, int order, int nthreads)
{
    // compute filter coefficients
    const float
    nsigma = sigma < 0.1f ? 0.1f : sigma,
    alpha = 1.695f / nsigma,
    ema = (float)std::exp(-alpha),
    ema2 = (float)std::exp(-2*alpha),
    b1 = -2*ema,
    b2 = ema2;

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;  
   
    const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
    a0 = k;
    a1 = k*(alpha-1)*ema;
    a2 = k*(alpha+1)*ema;
    a3 = -k*ema2;      
    coefp = (a0+a1)/(1+b1+b2);
    coefn = (a2+a3)/(1+b1+b2);

    // process columns
#if USE_SIMPLE_FILTER
    d_simpleRecursive_rgba<<< iDivUp(width, nthreads), nthreads >>>(d_src, d_temp, width, height, ema);
#else
    d_recursiveGaussian_rgba<<< iDivUp(width, nthreads), nthreads >>>(d_src, d_temp, width, height, a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif
    getLastCudaError("Kernel execution failed");

    transpose(d_temp, d_dest, width, height);
    getLastCudaError("transpose: Kernel execution failed");

    // process rows

    d_recursiveGaussian_rgba<<< iDivUp(height, nthreads), nthreads >>>(d_dest, d_temp, height, width, a0, a1, a2, a3, b1, b2, coefp, coefn);

    getLastCudaError("Kernel execution failed");

    transpose(d_temp, d_dest, height, width);
}
