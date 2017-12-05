
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <emmintrin.h>
#include <nmmintrin.h>


/**
 * Will perform a convolution on the input image
 * Original code from here:
 * http://www.songho.ca/dsp/convolution/convolution.html#cpp_conv2d
 */
void perform_convolution(double *kernel, int kRows, int kCols,
                            double *in, double *out, int rows, int cols) {

    // find center position of kernel (half of kernel size)
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;

    // Corrected locations to start filter on
    int mm,nn;
    int ii,jj;

    // rows
    for(int i=0; i < rows; ++i)
    {
        // columns
        for(int j=0; j < cols; ++j)
        {
            // kernel rows
            for(int m=0; m < kRows; ++m)
            {
                // row index of flipped kernel
                mm = kRows - 1 - m;
                // kernel columns
                for(int n=0; n < kCols; n+=2)
                {
                    // column index of flipped kernel
                    nn = kCols - 1 - n;
                    // index of input signal, used for checking boundary
                    ii = i + (m - kCenterY);
                    jj = j + (n - kCenterX);


                	if( ii >= 0 && ii < rows && jj+1 >= 0 && jj+1 < cols && nn+1 < kCols) {
	                    // calculate 2d => 1d mapping
	                    int tempout = i*rows + j;
	                    int tempin = ii*rows + j;
	                    int tempkerneel = mm*kRows + nn;
	                	// Load the next two doubles from both lists
	                    __m128d va = _mm_load_pd(&in[tempin]);
	                    __m128d vb = _mm_load_pd(&kernel[tempkerneel]);
	                    std::cout << va[0] << std::endl;
	                    std::cout << va[1] << std::endl;
	                	// multiple the two doubles
	                    __m128d vc = _mm_mul_pd(va,vb);
	                    // save the results
	                	//double out[2];
	                	//_mm_store_pd(out,vc);
	                    // ignore input samples which are out of bound
	                    // if( ii >= 0 && ii < rows && jj >= 0 && jj < cols ) {
                		double temp;
                		_mm_storeh_pd(&temp,vc);
                        out[tempout] += out[0];
                        std::cout << "got (1) = " << temp << std::endl;                	
                		//double temp;
                		_mm_storel_pd(&temp,vc);
                        out[tempout+1] += temp;
                        std::cout << "got (2) = " << temp << std::endl;
                    }
                }
            }
        }
    }
}



/**
 * Main method that will profile our convolution function
 */
int main(int argc, const char* argv[]) {

    // Size of our matrix and kernals
    const int imgSize = 300;
    const int kernelSize = 10;
    std::cout << "double = " << sizeof(double) << std::endl;

    // Allocate variables on stack
    double* imgIn = (double*)aligned_alloc(16,imgSize*imgSize*sizeof(double));
    double* imgOut = (double*)aligned_alloc(16,imgSize*imgSize*sizeof(double));
    double* kernel = (double*)aligned_alloc(16,kernelSize*kernelSize*sizeof(double)); 

    // Total time
    int loopCt = 100;
    double sumTime = 0.0;
    double* times = new double[loopCt];

    // Send it to our function!
    for(int i=0; i<loopCt; ++i) {

        // Generate a image matrices
        for(int i=0; i<imgSize; ++i) {
            for(int j=0; j<imgSize; ++j) {
                imgIn[i*imgSize+j] = 5;
                imgOut[i*imgSize+j] = 0;
            }
        }

        // Generate kernel
        for(int i=0; i<kernelSize; ++i) {
            for(int j=0; j<kernelSize; ++j) {
                kernel[i*kernelSize+j] = 1.0/9.0;
            }
        }

        // Run the code
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
        perform_convolution(kernel, kernelSize, kernelSize, imgIn, imgOut, imgSize, imgSize);
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        double runTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // Store our results
        times[i] = runTime;
        sumTime += runTime;
        //std::cout << "loop #" << i << " = " << runTime << std::endl;
    }

    // Print average
    std::cout << std::fixed << std::setprecision(2) << sumTime/loopCt << " ms average" << std::endl;

    // Calculate the std deviation
    double var = 0;
    for(int n=0; n<loopCt; ++n) {
      var += std::pow((times[n] - sumTime/loopCt),2);
    }
    var /= loopCt;
    double deviation = std::sqrt(var);
    std::cout << std::fixed << std::setprecision(2) << deviation << " sigma deviation"  << std::endl;

}
















