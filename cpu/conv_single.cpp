
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>


using namespace std;

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
                        temp += in[tempin] * kernel[tempkerneel];
                    }
                }
            }

            // Finally write to memory location
            out[tempout] = temp;

        }
    }
}



/**
 * Main method that will profile our convolution function
 */
int main(int argc, const char* argv[]) {

    // Check to make sure we get an matrix size
    if(argc < 2) {
        cerr << "Please specify a size of the image matrix" << endl;
        cerr << "./conv_single <imgsize>" << endl;
        return EXIT_FAILURE;
    }

    // Size of our matrix and kernals
    const int imgSize = std::atoi(argv[1]);
    const int kernelSize = 10;

    // Allocate variables on stack
    double* imgIn = new double[imgSize*imgSize];
    double* imgOut = new double[imgSize*imgSize];
    double* kernel = new double[kernelSize*kernelSize];    

    // Total time
    int loopCt = 25;
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
        //std::cout << "loop #" << i << " = " << runTime << " with " << imgOut[imgSize*imgSize/2] << std::endl;
    }

    // Print average
    std::cout << std::fixed << std::setprecision(4) << sumTime/loopCt << " ms average" << std::endl;

    // Calculate the std deviation
    double var = 0;
    for(int n=0; n<loopCt; ++n) {
      var += std::pow((times[n] - sumTime/loopCt),2);
    }
    var /= loopCt;
    double deviation = std::sqrt(var);
    std::cout << std::fixed << std::setprecision(4) << deviation << " sigma deviation"  << std::endl;

}






















