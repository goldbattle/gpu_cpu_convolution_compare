#ifndef CONV_KERNEL_H
#define CONV_KERNEL_H

__global__
void perform_convolution(double *kernel, int kRows, int kCols,
                            double *in, double *out, int rows, int cols);



#endif //CONV_KERNEL_H