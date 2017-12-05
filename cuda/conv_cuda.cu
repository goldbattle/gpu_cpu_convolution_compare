
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "conv_kernel.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace std;

/**
 * Main method that will profile our convolution function
 */
int main(int argc, const char* argv[]) {

    // Check to make sure we get an matrix size
    if(argc < 2) {
        cerr << "Please specify a size of the image matrix" << endl;
        cerr << "./conv_cuda <imgsize>" << endl;
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
    double sumTimeCopy1 = 0.0;
    double sumTimeKernel = 0.0;
    double sumTimeCopy2 = 0.0;
    double sumTimeFree = 0.0;
    double* times = new double[loopCt];

    // Startup the GPU device
    // https://devtalk.nvidia.com/default/topic/895513/cuda-programming-and-performance/cudamalloc-slow/post/4724457/#4724457
    cudaFree(0);

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
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        // Allocate memory on the device
        double* imgInD, *imgOutD, *kernelD;
        gpuErrchk(cudaMalloc((void**)&imgInD, imgSize*imgSize*sizeof(double)));
        gpuErrchk(cudaMalloc((void**)&imgOutD, imgSize*imgSize*sizeof(double)));
        gpuErrchk(cudaMalloc((void**)&kernelD, kernelSize*kernelSize*sizeof(double)));
        
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
        // Copy our data to the device
        gpuErrchk(cudaMemcpy(imgInD, imgIn, imgSize*imgSize*sizeof(double), cudaMemcpyHostToDevice));
        //gpuErrchk(cudaMemcpy(imgOutD, imgOut, imgSize*imgSize*sizeof(double), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(kernelD, kernel, kernelSize*kernelSize*sizeof(double), cudaMemcpyHostToDevice));

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        // Calculate size of kernel
        int block_size = 32;
        int grid_sizex = 0;
        int grid_sizey = 0;
        int size_totalx = imgSize;
        int size_totaly = imgSize;

        //Calculate grid_size (add 1 if not evenly divided)
        if(size_totalx%block_size == 0) {
            grid_sizex = ceil(size_totalx/block_size);
        } else {
            grid_sizex = ceil(size_totalx/block_size) + 1;
        }
        if(size_totaly%block_size == 0) {
            grid_sizey = ceil(size_totaly/block_size);
        } else {
            grid_sizey = ceil(size_totaly/block_size) + 1;
        }

        // Create size objects
        dim3 DimGrid(grid_sizex,grid_sizey,1);
        dim3 DimBlock(block_size,block_size,1);

        // Debug
        //cout << "grid_size = " << grid_size << endl;
        //cout << "block_size = " << block_size << endl;

        // Launch the kernel
        perform_convolution<<<DimGrid, DimBlock>>>(kernelD,kernelSize,kernelSize,imgInD,imgOutD,imgSize,imgSize);

        // Sync after the kernel is launched
        cudaDeviceSynchronize();

        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

        // Copy back to host
        gpuErrchk(cudaMemcpy(imgOut, imgOutD, imgSize*imgSize*sizeof(double), cudaMemcpyDeviceToHost));

        
        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();

        // Free the data
        gpuErrchk(cudaFree(imgInD));
        gpuErrchk(cudaFree(kernelD));
        gpuErrchk(cudaFree(imgOutD));


        std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
        double runTime = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t0).count();

        // Store our results
        times[i] = runTime;
        sumTime += runTime;
        sumTimeCopy1 += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        sumTimeKernel += std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        sumTimeCopy2 += std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
        sumTimeFree += std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
        // std::cout << "loop #" << i << " = " << runTime << " with " << imgOut[10] << " "  << imgOut[1000] << std::endl;
        // std::cout << "\tmalloc: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << std::endl;
        // std::cout << "\tcopy: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
        // std::cout << "\tkernel: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << std::endl;
        // std::cout << "\tcopy: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << std::endl;
        // std::cout << "\tfree: " << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << std::endl;
    }

    // Print average
    std::cout << std::fixed << std::setprecision(5) << sumTime/loopCt << " ms average" << std::endl;

    // Calculate the std deviation
    double var = 0;
    for(int n=0; n<loopCt; ++n) {
      var += std::pow((times[n] - sumTime/loopCt),2);
    }
    var /= loopCt;
    double deviation = std::sqrt(var);
    std::cout << std::fixed << std::setprecision(5) << deviation << " sigma deviation"  << std::endl;

    // Extra times for GPU computing
    std::cout << "\t" << std::fixed << std::setprecision(5) << sumTimeCopy1/loopCt << " ms average (copy1)" << std::endl;
    std::cout << "\t" << std::fixed << std::setprecision(5) << sumTimeKernel/loopCt << " ms average (kernel)" << std::endl;
    std::cout << "\t" << std::fixed << std::setprecision(5) << sumTimeCopy2/loopCt << " ms average (copy2)" << std::endl;
    std::cout << "\t" << std::fixed << std::setprecision(5) << sumTimeFree/loopCt << " ms average (free)" << std::endl;

}






















