
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include "conv_kernel.h"

/**
 * Will perform a convolution on the input image
 * Original code from here:
 * http://www.songho.ca/dsp/convolution/convolution.html#cpp_conv2d
 */
__global__
void perform_convolution(double *kernel, int kRows, int kCols,
                            double *in, double *out, int rows, int cols) {

    // For now a hack
    assert(kRows == 10);
    assert(kCols == 10);

    // Copy to shared memory
    __shared__ double skernel[10*10];

    if (threadIdx.x < 10 && threadIdx.y < 10) {
          skernel[threadIdx.x*kRows+threadIdx.y] = kernel[threadIdx.x*kRows+threadIdx.y];
    }

    // Sync so we know we have copied everything
    __syncthreads();

    // find center position of kernel (half of kernel size)
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;

    // Corrected locations to start filter on
    int mm,nn;
    int ii,jj;

    // Calculate our given location from our block/thread id
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    // Return if we are out of bounds
    if(i > rows || j > cols)
        return;

    // Temp variable to store the sum in
    int tempout = i*rows + j;
    double temp = 0.0;

    // kernel rows
    for(int m=0; m < kRows; ++m)
    {
        // row index of flipped kernel
        mm = kRows - 1 - m;
        // kernel columns
        for(int n=0; n < kCols; ++n)
        {
            // column index of flipped kernel
            nn = kCols - 1 - n;
            // index of input signal, used for checking boundary
            ii = i + (m - kCenterY);
            jj = j + (n - kCenterX);
            // ignore input samples which are out of bound
            if( ii >= 0 && ii < rows && jj >= 0 && jj < cols ) {
                // calculate 2d => 1d mapping
                int tempin = ii*rows + j;
                int tempkerneel = mm*kRows + nn;
                // multiple it times our kernel
                temp += in[tempin] * skernel[tempkerneel];
            }
        }
    }

    // Finally write to memory location
    out[tempout] = temp;
        
    
}