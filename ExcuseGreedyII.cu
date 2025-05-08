/*
  Version 1.0, 08.06.2015, Copyright University of Tübingen.
  The Code is created based on the method from the paper:
  "ExCuSe: Robust Pupil Detection in Real-World Scenarios", W. Fuhl, T. C. Kübler, K. Sippel, W. Rosenstiel, E. Kasneci
  CAIP 2015 : Computer Analysis of Images and Patterns
  The code and the algorithm are for non-comercial use only.

   The code is parallelized using CUDA and CUBLAS by Arturo Vicente Jaén. 05/04/2023. Copyrigth University of Murcia.
   This file contains the CUDA kernels and functions for the pupil detection working together for the parallelized final version.
*/
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/detail/config.h>
#include <thrust/reduce.h>
#include <cusolverDn.h>
#include <thrust/system/cuda/execution_policy.h>
#include <fstream>
#include <map>

#include "include/Pupil.h"

using namespace std;
using namespace cv;

namespace ExcuseGreedyII {
    int max_ellipse_radi = 50;
    int good_ellipse_threshold = 15;


    #define MAX_LINE 4
    #define MAX_LINE_CPU 10000
    #define IMG_SIZE 1100
    #define DEF_SIZE 1000
    #define KERNEL_SIZE 16
    #define HALF_KERNEL_SIZE 8
    #define MAX_CURVES 1500

    //--------------------------------------------- VARIABLE DEFINITIONS -------------------------------------------------

    // Constant memory
    __device__ __constant__ int _nmsPrecalc[8][8];
    __device__ __constant__ int _lowanglePrecalc[2][8];
    __device__ __constant__ float _gauC[16];
    __device__ __constant__ float _deriv_gauC[16];
    __device__ __constant__ int _raysPrecalc[2][8];

    // Global memory
    unsigned int *d_grayHist, *h_grayHist, *d_innerGray, *d_outputImg, *d_translation, *d_sum_x, *d_sum_y, *d_total, *d_min;
    float *d_meanFeld, *d_stdFeld, *d_smallPic, *d_auxSmall, *d_aux2Small, *d_resX, *d_resY, *d_strong, *d_weak,
            *d_exitSmall, *d_a, *d_a2inv, **d_array, *d_b, *d_x, *h_x;
    unsigned char *d_pic, *d_edges, *d_edgesAux, *d_th_edges;
    bool *d_excentricity;
    int *d_xx, *d_yy, *d_info, *d_hist_l, *d_hist_lb, *d_hist_b, *d_hist_br,
            *h_hist_l, *h_hist_lb, *h_hist_b, *h_hist_br, *d_ret;

    // Cublas handle
    cublasHandle_t d_handlePeek, d_handle;

    // Cuda streams
    cudaStream_t d_stream_1, d_stream_2;

    // Cuda symbols directions
    float *_d_sum_y, *_d_sum_x, *_translationIndex, *_atomicInnerGrayIndex, *_atomicIndexRaysPoints, *_min_val, *_pos_x, *_pos_y, *_pos_count;

    //--------------------------------------------- CANNY KERNELS -------------------------------------------------

    /**
    * @brief kernel for applying a gaussian 1D filter by rows (in the middle of the kernel as anchor (-1,-1))
    * @param src source matrix
    * @param dst destination matrix
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void gau1Drow(float *src, float *dst, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;

        if (i < height && j < width) {
            float sum = 0;
    #pragma unroll
            for (int yprim = 0; yprim < KERNEL_SIZE; yprim++) {
                int jj = min(max(j + yprim - HALF_KERNEL_SIZE, 0), width - 1);
                sum += _gauC[yprim] * src[i * width + jj];
            }
            dst[i * width + j] = sum;
        }
    }

    /**
    * @brief kernel for applying a gaussian 1D filter by columns (in the middle of the kernel as anchor (-1,-1))
    * @param src source matrix
    * @param dst destination matrix
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void gau1Dcol(float *src, float *dst, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;

        if (i < height && j < width) {
            float sum = 0;
    #pragma unroll
            for (int yprim = 0; yprim < KERNEL_SIZE; yprim++) {
                int ii = min(max(i + yprim - HALF_KERNEL_SIZE, 0), height - 1);
                sum += _gauC[yprim] * src[ii * width + j];
            }
            dst[i * width + j] = sum;
        }
    }

    /**
    * @brief kernel for applying a deriv gau (Sobel) 1D filter by rows (in the middle of the kernel as anchor (-1,-1))
    * @param src source matrix
    * @param dst destination matrix
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void deriv1Drow(float *src, float *dst, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;

        if (i < height && j < width) {
            float sum = 0;
    #pragma unroll
            for (int yprim = 0; yprim < KERNEL_SIZE; yprim++) {
                int jj = min(max(j + yprim - HALF_KERNEL_SIZE, 0), width - 1);
                sum += _deriv_gauC[yprim] * src[i * width + jj];
            }
            dst[i * width + j] = sum;
        }
    }

    /**
    * @brief kernel for applying a deriv gau (Sobel) 1D filter by columns (in the middle of the kernel as anchor (-1,-1))
    * @param src source matrix
    * @param dst destination matrix
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void deriv1Dcol(float *src, float *dst, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;

        if (i < height && j < width) {
            float sum = 0;
    #pragma unroll
            for (int yprim = 0; yprim < KERNEL_SIZE; yprim++) {
                int ii = min(max(i + yprim - HALF_KERNEL_SIZE, 0), height - 1);
                sum += _deriv_gauC[yprim] * src[ii * width + j];
            }
            dst[i * width + j] = sum;
        }
    }

    /**
    * @brief kernel for calculating the hypotenuse of two values
    * @param res_x gradient in x direction
    * @param res_y gradient in y direction
    * @param dst destination matrix
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void hypot(float *res_x, float *res_y, float *dst, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;

        if (i < height && j < width) {
            int idx = i * width + j;
            dst[idx] = hypotf(res_x[idx], res_y[idx]);
    //        int val = __float2int_ru(dst[idx]);
        }
    }

    /**
    * @brief kernel for normalizing a matrix
    * @param src source matrix
    * @param dst destination matrix
    * @param width number of columns
    * @param height number of rows
    * @param alpha lower next bound
    * @param beta upper next bound
    * @param min minimum value of the current matrix
    * @param max maximum value of the current matrix
    */
    __global__
    void normalize(float *src, float *dst, int width, int height, float alpha, float beta, float min, float max) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;

        if (i < height && j < width) {
            int idx = i * width + j;
            dst[idx] = (src[idx] - min) / (max - min) * (beta - alpha) + alpha;
        }
    }

    __device__ float high_th;

    /**
    * @brief kernel for non-maxima suppression
    * @param res gradient matrix
    * @param res_x gradient in x direction
    * @param res_y gradient in y direction
    * @param strong strong edges matrix
    * @param weak weak edges matrix
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void nonMaximaSuppresion(float *res, float *res_x, float *res_y, float *strong, float *weak, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;

        if (i < height - 1 && i > 1 && j < width - 1 && j > 1) {
            float ix = res_x[i * width + j];
            float iy = res_y[i * width + j];
            float angle = atan2f(iy, ix) / CUDART_PI_F + 1;
            int angle_idx = __float2int_rd(angle * 4);
            int despi1grad1 = _nmsPrecalc[angle_idx][0];
            int despj1grad1 = _nmsPrecalc[angle_idx][1];
            int despi2grad1 = _nmsPrecalc[angle_idx][2];
            int despj2grad1 = _nmsPrecalc[angle_idx][3];
            int despi1grad2 = _nmsPrecalc[angle_idx][4];
            int despj1grad2 = _nmsPrecalc[angle_idx][5];
            int despi2grad2 = _nmsPrecalc[angle_idx][6];
            int despj2grad2 = _nmsPrecalc[angle_idx][7];
            float d = abs(iy / ix) * (angle_idx == 0 || angle_idx == 3 || angle_idx == 4 || angle_idx == 7) +
                    abs(ix / iy) * (angle_idx == 1 || angle_idx == 2 || angle_idx == 5 || angle_idx == 6);
            float grad1 = (res[(i + despi1grad1) * width + j + despj1grad1] * (1 - d)) +
                        (res[(i + despi2grad1) * width + j + despj2grad1] * d);
            float grad2 = (res[(i + despi1grad2) * width + j + despj1grad2] * (1 - d)) +
                        (res[(i + despi2grad2) * width + j + despj2grad2] * d);
            weak[i * width + j] = (res[i * width + j] > grad1 && res[i * width + j] > grad2) * 255;
            strong[i * width + j] = (weak[i * width + j] == 255 && res[i * width + j] > high_th) * 255;
        }
    }

    /**
    * @brief function for managing full circular queue
    * @param front
    * @param back
    * @return true if the queue is full
    */
    __device__ bool isFull(int &front, int &back) {
        return (front == 0 && back == 9) || (front == back + 1);
    }

    /**
    * @brief function for managing size of circular queue
    * @param front
    * @param back
    * @return size of the queue
    */
    __device__ int getSize(int &front, int &back) {
        if (front == -1) {
            return 0;
        } else if (back >= front) {
            return back - front + 1;
        } else {
            return (back + 1) + (9 - front + 1);
        }
    }

    /**
    * @brief function for managing empty circular queue
    * @param front
    * @param back
    * @return true if the queue is empty
    */
    __device__ bool isEmpty(int &front, int &back) {
        return front == -1;
    }

    /**
    * @brief function for pushing elements into circular queue
    * @param front
    * @param back
    * @param vec circular queue
    * @param i value to push
    * @param j value to push
    */
    __device__ void push_back(int &front, int &back, thrust::pair<int, int> *vec, int i, int j) {
        if (isFull(front, back)) {
            return;
        } else if (front == -1) {
            front = 0;
            back = 0;
            vec[back] = thrust::make_pair(i, j);
        } else if (back == 9 && front != 0) {
            back = 0;
            vec[back] = thrust::make_pair(i, j);
        } else {
            back++;
            vec[back] = thrust::make_pair(i, j);
        }

    }

    /**
    * @brief function for popping elements from circular queue
    * @param front
    * @param back
    * @param vec circular queue
    * @return pair of values
    */
    __device__ thrust::pair<int, int> pop_back(int &front, int &back, thrust::pair<int, int> *vec) {
        if (isEmpty(front, back)) {
            return;
        }
        thrust::pair<int, int> ret = vec[front];
        if (front == back) {
            front = -1;
            back = -1;
        } else if (front == 9) {
            front = 0;
        } else {
            front++;
        }
        return ret;
    }

    /**
    * @brief function for hysteresis thresholding
    * @param strong strong edges matrix
    * @param weak weak edges matrix
    * @param exit output matrix
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void hysteresis_pro(float *strong, float *weak, float *exit, int width, int height) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        // Define circular queue
        thrust::pair<int, int> vec[10];
        int front = -1;
        int back = -1;

        if (i < height && j < width) {
            if (strong[i * width + j] == 255) {
                push_back(front, back, vec, i, j);
                int limit = 0;
                while (!isEmpty(front, back) && limit < MAX_LINE) {
                    thrust::pair<int, int> d = pop_back(front, back, vec);
                    int ii = d.first;
                    int jj = d.second;
                    if (exit[ii * width + jj] != 255) {
                        exit[ii * width + jj] = 255;
    #pragma unroll
                        for (int k = 0; k < 8; k++) {
                            int k1 = _lowanglePrecalc[0][k];
                            int k2 = _lowanglePrecalc[1][k];
                            int idx = (ii + k2) * width + jj + k1;
                            if (weak[idx] == 255 && strong[idx] != 255 && exit[idx] != 255) {
                                push_back(front, back, vec, ii + k2, jj + k1);
                                limit++;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
    * @brief kernel for calculating point of the historgram with 70% of pixels in the left side
    * @param hist histogram
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void operateHist(unsigned int *hist, int width, int height) {
        int sum = 0;
        high_th = 7;
        float PercentOfPixelsNotEdges = 0.7 * width * height;

    #pragma unroll
        for (int i = 0; i < 64; i++) {
            sum += hist[i];
            if (sum > PercentOfPixelsNotEdges) {
                high_th = float(i + 1) / float(64);
                break;
            }
        }
    }

    /**
    * @brief kernel for calculating histogram without using shared memory (for 32 float images)
    * @param src  input matrix
    * @param hist output matrix
    * @param width number of columns
    * @param height number of rows
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    */
    __global__
    void
    calcHistInt(float *src, unsigned int *hist, int width, int height, int start_x, int end_x, int start_y, int end_y) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / (end_x - start_x);
        int j = index % (end_x - start_x);
        if (i < (end_y - start_y)) {
            int idx = __float2ull_ru(src[(i + start_y) * width + (j + start_x)]);
            atomicAdd(&hist[idx], 1);
        }
    }

    //---------------------------------------------- PEEK KERNELS ----------------------------------------------------------------------------

    /**
    * @brief kernel for calculating histogram using shared memory (for 8 bit images)
    * @param src input matrix (in 8 bit format)
    * @param hist output histogram
    * @param width number of columns
    * @param height number of rows
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    */
    __global__ void
    histAtomic(unsigned char *src, unsigned int *hist, int width, int height, int start_x, int end_x, int start_y,
            int end_y) {
        // pixel coordinates
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // grid dimensions
        int nx = blockDim.x * gridDim.x;
        int ny = blockDim.y * gridDim.y;

        // linear thread index within 2D block
        int t = threadIdx.x + threadIdx.y * blockDim.x;

        // total threads in 2D block
        int nt = blockDim.x * blockDim.y;

        // initialize temporary accumulation array in shared memory
        extern __shared__ unsigned int smem[];
        for (int i = t; i < 256; i += nt) smem[i] = 0;
        __syncthreads();

        // process pixels
        // updates our block's partial histogram in shared memory
        for (int col = x; col < (end_x - start_x); col += nx)
            for (int row = y; row < (end_y - start_y); row += ny) {
                unsigned int r = src[(row + start_y) * width + col + start_x];
                atomicAdd(&smem[r], 1);
            }
        __syncthreads();

        // write partial histogram into the global memory
        for (int i = t; i < 256; i += nt) {
            atomicAdd(&hist[i], smem[i]);
        }
    }

    /**
    * @brief kernel for calculating mean feld
    * @param src source image
    * @param mean mean feld
    * @param width number of columns
    * @param height number of rows
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    */
    __global__ void
    calcMeanFeld(unsigned char *src, float *mean, int width, int height, int start_x, int end_x, int start_y, int end_y) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / (end_x - start_x);
        int j = index % (end_x - start_x);
        if (i < (end_y - start_y)) {
            float idxflt = static_cast<float>(src[(i + start_y) * width + (j + start_x)]);
            atomicAdd(&mean[j], idxflt);
        }
    }

    /**
    * @brief kernel for calculating mean division
    * @param mean mean feld
    * @param width number of columns
    * @param height number of rows

    */
    __global__
    void calcMeanDiv(float *mean, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < width) {
            mean[index] = fdividef(mean[index], (float) height);
        }
    }

    /**
    * @brief kernel for calculating standard deviation
    * @param src source image
    * @param stdDev standard deviation
    * @param mean mean feld
    * @param width number of columns
    * @param height number of rows
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    */
    __global__
    void
    calcStdDev(unsigned char *src, float *stdDev, float *mean, int width, int height, int start_x, int end_x, int start_y,
            int end_y) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / (end_x - start_x);
        int j = index % (end_x - start_x);
        if (i < (end_y - start_y)) {
            int idx = (int) (src[(i + start_y) * width + (j + start_x)]);
            atomicAdd(&stdDev[j], (idx - mean[j]) * (idx - mean[j]));
        }
    }

    /**
    * @brief kernel for calculating standard deviation division
    * @param stdDev standard deviation
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void calcStdDevDiv(float *stdDev, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < width) {
            stdDev[index] = sqrtf(fdividef(stdDev[index], (float) height));
        }
    }

    //-------------------------------------------------- BORDER KERNELS ------------------------------------------------------------------------

    /**
    * @brief kernel for removing borders (also performs type conversion byte to float)
    * @param src source image
    * @param dst destination image
    * @param width number of columns
    * @param height number of rows
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    */
    __global__
    void removeBorders(unsigned char *src, float *dst, int width, int height, int start_x, int end_x, int start_y,
                    int end_y) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);
        if (j < (end_y - start_y)) {
            dst[j * (end_x - start_x) + i] = float(src[(j + start_y) * width + (i + start_x)]);
        }
    }

    /**
    * @brief kernel for adding borders (also performs type conversion float to byte)
    * @param src source image
    * @param dst destination image
    * @param width number of columns
    * @param height number of rows
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    */
    __global__
    void addBorders(float *src, unsigned char *dst, int width, int height, int start_x, int end_x, int start_y,
                    int end_y) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);
        if (j < (end_y - start_y)) {
            dst[(j + start_y) * width + (i + start_x)] = char(__float2int_ru(src[j * (end_x - start_x) + i]));
        }
    }

    //----------------------------------------- REMOVE POINTS KERNELS ---------------------------------------------------------------------------------

    /**
    * @brief kernel for removing points with low angle in the first step, which removes points with low angle and less than 3 neighbors
    * @param src source image
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void remove_points_with_low_angle_gpu_1(unsigned char *src, int start_x, int end_x, int start_y, int end_y, int width,
                                            int height) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        thrust::pair<int, int> vec[10];
        int front = -1;
        int back = -1;

        if (j < (end_y - start_y)) {
            j += start_y;
            i += start_x;
            int idx = j * width + i;
            if ((int) src[idx]) {
                push_back(front, back, vec, i, j);
                int limit = 0;
                while (!isEmpty(front, back) && limit < MAX_LINE) {
                    thrust::pair<int, int> d = pop_back(front, back, vec);
                    int ii = d.first;
                    int jj = d.second;
                    int box[8];
                    box[0] = (int) src[(jj - 1) * width + (ii - 1)];
                    box[1] = (int) src[(jj - 1) * width + (ii)];
                    box[2] = (int) src[(jj - 1) * width + (ii + 1)];
                    box[3] = (int) src[(jj) * width + (ii + 1)];
                    box[4] = (int) src[(jj + 1) * width + (ii + 1)];
                    box[5] = (int) src[(jj + 1) * width + (ii)];
                    box[6] = (int) src[(jj + 1) * width + (ii - 1)];
                    box[7] = (int) src[(jj) * width + (ii - 1)];
                    bool valid = false;
    # pragma unroll
                    for (int k = 0; k < 8 && !valid; k++)
                        valid = (box[k] && (box[(k + 2) % 8] || box[(k + 3) % 8] || box[(k + 4) % 8] || box[(k + 5) % 8] ||
                                            box[(k + 6) % 8]));
                    if (!valid) {
                        src[jj * width + ii] = 0;
    #pragma unroll
                        for (int k = 4; k < 8; k++) {
                            int iii = ii + _lowanglePrecalc[0][k];
                            int jjj = jj + _lowanglePrecalc[1][k];
                            if (src[jjj * width + iii]) {
                                push_back(front, back, vec, iii, jjj);
                                limit++;
                            }
                        }
                    }
                }
            }
        }
    }


    /**
    * @brief kernel for removing points with low angle in the second step (Uses just diagonal parallelism)
    * @param src source image
    * @param dst destination image
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param width number of columns
    * @param height number of rows
    */
    __global__ void
    remove_points_with_low_angle_gpu_2_diagonal(unsigned char *src, unsigned char *dst, int start_x, int end_x, int start_y,
                                                int end_y, int width, int height)  {
        int index_real = threadIdx.x;
        int numDiag = width + height - 1;
        int totalDiagSize = min(width, height);
        for (int k = 0; k < numDiag; k++) {
            for (int index = index_real; index < totalDiagSize; index += blockDim.x)
            {
                int i = index;
                int j = k - index;
                if (i < (end_x - start_x) && j < (end_y - start_y) && j >= 0) {
                    j += start_y;
                    i += start_x;
                    int idx = j * width + i;
                    if ((int)src[idx]) {
                        int N, S, E, W;
                        N = (int)src[(j - 1) * width + (i)];
                        W = (int)src[(j)*width + (i - 1)];
                        E = (int)src[(j)*width + (i + 1)];
                        S = (int)src[(j + 1) * width + (i)];
                        if ((N && E) || (N && W) || (S && E) || (S && W)) {
                            dst[idx] = 0;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }


    /**
    * @brief kernel for removing points with low angle in the second step (first phase) which aplies "L" morphological operation in one diagonal
    * @param src source image
    * @param dst destination image
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void remove_points_with_low_angle_gpu_22(unsigned char *src, unsigned char *dst, int start_x, int end_x, int start_y,
                                            int end_y, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);
        if (j < (end_y - start_y)) {
            j += start_y;
            i += start_x;
            int idx = j * width + i;
            if ((int) src[idx]) {
                int box[4];
                box[0] = (int) src[(j - 1) * width + (i)];
                box[1] = (int) src[(j) * width + (i - 1)];
                box[2] = (int) src[(j) * width + (i + 1)];
                box[3] = (int) src[(j + 1) * width + (i)];

                dst[(j) * width + (i)] *= !((box[2] && box[3]) || (box[0] && box[2]));
            }
        }
    }

    /**
    * @brief kernel for removing points with low angle in the second step (second phase) which aplies "L" morphological operation in the other diagonal
    * @param src source image
    * @param dst destination image
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void remove_points_with_low_angle_gpu_21(unsigned char *src, unsigned char *dst, int start_x, int end_x, int start_y,
                                            int end_y, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);
        if (j < (end_y - start_y)) {
            j += start_y;
            i += start_x;
            int idx = j * width + i;
            if ((int) src[idx]) {
                int box[4];
                box[0] = (int) src[(j - 1) * width + (i)];
                box[1] = (int) src[(j) * width + (i - 1)];
                box[2] = (int) src[(j) * width + (i + 1)];
                box[3] = (int) src[(j + 1) * width + (i)];

                dst[(j) * width + (i)] *= !((box[1] && box[3]) || (box[0] && box[1]));
            }
        }
    }

    /**
    * @brief kernel for applying third morphological operation in the second step
    * @param src source image
    * @param dst destination image
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void remove_points_with_low_angle_gpu_3(unsigned char *src, unsigned char *dst, int start_x, int end_x, int start_y,
                                            int end_y, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);
        if (j < (end_y - start_y)) {
            j += start_y;
            i += start_x;
            int idx = j * width + i;
            if ((int) src[idx] == 255) {
                int box[17];
                box[0] = (int) src[(j - 1) * width + (i - 1)];
                box[1] = (int) src[(j - 1) * width + (i)];
                box[2] = (int) src[(j - 1) * width + (i + 1)];

                box[3] = (int) src[(j) * width + (i - 1)];
                box[4] = (int) src[(j) * width + (i)];
                box[5] = (int) src[(j) * width + (i + 1)];

                box[6] = (int) src[(j + 1) * width + (i - 1)];
                box[7] = (int) src[(j + 1) * width + (i)];
                box[8] = (int) src[(j + 1) * width + (i + 1)];
                //external
                box[9] = (int) src[(j) * width + (i + 2)];
                box[10] = (int) src[(j + 2) * width + (i)];

                box[11] = (int) src[(j) * width + (i + 3)];
                box[12] = (int) src[(j - 1) * width + (i + 2)];
                box[13] = (int) src[(j + 1) * width + (i + 2)];

                box[14] = (int) src[(j + 3) * width + (i)];
                box[15] = (int) src[(j + 2) * width + (i - 1)];
                box[16] = (int) src[(j + 2) * width + (i + 1)];


                bool expresion1 = ((box[10] && !box[7]) && (box[8] || box[6]));
                dst[(j + 1) * width + (i - 1)] *= !expresion1;
                dst[(j + 1) * width + (i + 1)] *= !expresion1;
    //            dst[(j + 1) * width + (i)] += expresion1 * 255;
                if (expresion1) {
                    dst[(j + 1) * width + (i)] = 255;
                }

                bool expresion2 = ((box[14] && !box[7] && !box[10]) && ((box[8] || box[6]) && (box[16] || box[15])));
                dst[(j + 1) * width + (i + 1)] *= !expresion2;
                dst[(j + 1) * width + (i - 1)] *= !expresion2;
                dst[(j + 2) * width + (i + 1)] *= !expresion2;
                dst[(j + 2) * width + (i - 1)] *= !expresion2;
    //            dst[(j + 1) * width + (i)] += expresion2 * 255;
    //            dst[(j + 2) * width + (i)] += expresion2 * 255;
                if (expresion2) {
                    dst[(j + 1) * width + (i)] = 255;
                    dst[(j + 2) * width + (i)] = 255;
                }

                bool expresion3 = ((box[9] && !box[5]) && (box[8] || box[2]));
                dst[(j + 1) * width + (i + 1)] *= !expresion3;
                dst[(j - 1) * width + (i + 1)] *= !expresion3;
    //            dst[(j) * width + (i + 1)] += expresion3 * 255;
                if (expresion3) {
                    dst[(j) * width + (i + 1)] = 255;
                }

                bool expresion4 = ((box[11] && !box[5] && !box[9]) && ((box[8] || box[2]) && (box[13] || box[12])));
                dst[(j + 1) * width + (i + 1)] *= !expresion4;
                dst[(j - 1) * width + (i + 1)] *= !expresion4;
                dst[(j + 1) * width + (i + 2)] *= !expresion4;
                dst[(j - 1) * width + (i + 2)] *= !expresion4;
    //            dst[(j) * width + (i + 1)] += expresion4 * 255;
    //            dst[(j) * width + (i + 2)] += expresion4 * 255;
                if (expresion4) {
                    dst[(j) * width + (i + 1)] = 255;
                    dst[(j) * width + (i + 2)] = 255;
                }
            }
        }
    }

    /**
    * @brief kernel for applying fourth morphological operation in the second step
    * @param src source image
    * @param dst destination image
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void remove_points_with_low_angle_gpu_4(unsigned char *src, unsigned char *dst, int start_x, int end_x, int start_y,
                                            int end_y, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);
        if (j < (end_y - start_y)) {
            j += start_y;
            i += start_x;
            int idx = j * width + i;
            if ((int) src[idx] == 255) {
                int box[33];
                box[0] = (int) src[(j - 1) * width + (i - 1)];
                box[1] = (int) src[(j - 1) * width + (i)];
                box[2] = (int) src[(j - 1) * width + (i + 1)];

                box[3] = (int) src[(j) * width + (i - 1)];
                box[4] = (int) src[(j) * width + (i)];
                box[5] = (int) src[(j) * width + (i + 1)];

                box[6] = (int) src[(j + 1) * width + (i - 1)];
                box[7] = (int) src[(j + 1) * width + (i)];
                box[8] = (int) src[(j + 1) * width + (i + 1)];

                box[9] = (int) src[(j - 1) * width + (i + 2)];
                box[10] = (int) src[(j - 1) * width + (i - 2)];
                box[11] = (int) src[(j + 1) * width + (i + 2)];
                box[12] = (int) src[(j + 1) * width + (i - 2)];

                box[13] = (int) src[(j - 2) * width + (i - 1)];
                box[14] = (int) src[(j - 2) * width + (i + 1)];
                box[15] = (int) src[(j + 2) * width + (i - 1)];
                box[16] = (int) src[(j + 2) * width + (i + 1)];

                box[17] = (int) src[(j - 3) * width + (i - 1)];
                box[18] = (int) src[(j - 3) * width + (i + 1)];
                box[19] = (int) src[(j + 3) * width + (i - 1)];
                box[20] = (int) src[(j + 3) * width + (i + 1)];

                box[21] = (int) src[(j + 1) * width + (i + 3)];
                box[22] = (int) src[(j + 1) * width + (i - 3)];
                box[23] = (int) src[(j - 1) * width + (i + 3)];
                box[24] = (int) src[(j - 1) * width + (i - 3)];

                box[25] = (int) src[(j - 2) * width + (i - 2)];
                box[26] = (int) src[(j + 2) * width + (i + 2)];
                box[27] = (int) src[(j - 2) * width + (i + 2)];
                box[28] = (int) src[(j + 2) * width + (i - 2)];

                box[29] = (int) src[(j - 3) * width + (i - 3)];
                box[30] = (int) src[(j + 3) * width + (i + 3)];
                box[31] = (int) src[(j - 3) * width + (i + 3)];
                box[32] = (int) src[(j + 3) * width + (i - 3)];

    //            dst[j * width + i] *= !(box[7] && box[2] && box[9]);
    //            dst[j * width + i] *= !(box[7] && box[0] && box[10]);
    //            dst[j * width + i] *= !(box[1] && box[8] && box[11]);
    //            dst[j * width + i] *= !(box[1] && box[6] && box[12]);
    //
    //            dst[j * width + i] *= !(box[0] && box[13] && box[17] && box[8] && box[11] && box[21]);
    //            dst[j * width + i] *= !(box[2] && box[14] && box[18] && box[6] && box[12] && box[22]);
    //            dst[j * width + i] *= !(box[6] && box[15] && box[19] && box[2] && box[9] && box[23]);
    //            dst[j * width + i] *= !(box[8] && box[16] && box[20] && box[0] && box[10] && box[24]);
    //
    //            dst[j * width + i] *= !(box[0] && box[25] && box[29] && box[2] && box[27] && box[31]);
    //            dst[j * width + i] *= !(box[0] && box[25] && box[29] && box[6] && box[28] && box[32]);
    //            dst[j * width + i] *= !(box[8] && box[26] && box[30] && box[2] && box[27] && box[31]);
    //            dst[j * width + i] *= !(box[8] && box[26] && box[30] && box[6] && box[28] && box[32]);

                if (box[7] && box[2] && box[9] || box[7] && box[0] && box[10] || box[1] && box[8] && box[11] ||
                    box[1] && box[6] && box[12] || box[0] && box[13] && box[17] && box[8] && box[11] && box[21] ||
                    box[2] && box[14] && box[18] && box[6] && box[12] && box[22] ||
                    box[6] && box[15] && box[19] && box[2] && box[9] && box[23] ||
                    box[8] && box[16] && box[20] && box[0] && box[10] && box[24] ||
                    box[0] && box[25] && box[29] && box[2] && box[27] && box[31] ||
                    box[0] && box[25] && box[29] && box[6] && box[28] && box[32] ||
                    box[8] && box[26] && box[30] && box[2] && box[27] && box[31] ||
                    box[8] && box[26] && box[30] && box[6] && box[28] && box[32]) {
                    dst[j * width + i] = 0;
                }
            }
        }
    }

    //-------------------------------------- ANGULAR HISTOGRAM KERNELS ------------------------------------------------------

    /**
    * @brief kernel for calculating the angular histograms in the 4 directions
    * @param src source image
    * @param hist_l left histogram
    * @param hist_b bottom histogram
    * @param hist_lb left-bottom histogram
    * @param hist_br bottom-right histogram
    * @param width image width
    * @param height image height
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    *
    */
    __global__ void
    calculateRotatedHists(unsigned char *src, int *hist_l, int *hist_b, int *hist_lb, int *hist_br,
                        int width, int height, int start_x, int end_x, int start_y, int end_y, float th) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);
        if (j < (end_y - start_y)) {
            j += start_y;
            i += start_x;
            if ((int) src[(width * j) + i] < th) {
                int idx_lb = (width / 2) + (i - (width / 2)) + (j);
                int idx_br = (width / 2) + (i - (width / 2)) + (height - j);
                if (j >= 0 && j < DEF_SIZE && i >= 0 && i < DEF_SIZE && idx_lb >= 0 && idx_lb < DEF_SIZE &&
                    idx_br >= 0 && idx_br < DEF_SIZE) {
                    atomicAdd(&hist_l[j], 1);
                    atomicAdd(&hist_b[i], 1);
                    atomicAdd(&hist_lb[idx_lb], 1);
                    atomicAdd(&hist_br[idx_br], 1);
                }
            }
        }
    }

    //---------------------------------------- FIT ELLIPSE KERNELS ----------------------------------------------------

    /**
    * @brief kernel for performing a per column sum
    * @param matrix matrix to sum
    * @param result vector to store the results
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void sumColumns(float *matrix, float *result, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;
        if (i < height && j < width) {
            atomicAdd(&result[j], matrix[i * width + j]);
        }
    }

    __device__ int sum_y = 0;
    __device__ int sum_x = 0;

    /**
    * @brief kernel for calculating the sum of the points (in order to calculate the mean)
    * @param x x coordinates
    * @param y y coordinates
    * @param size number of points
    */
    __global__
    void sumPoints(int *x, int *y, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            atomicAdd(&sum_x, x[index]);
            atomicAdd(&sum_y, y[index]);
        }
    }

    /**
    * @brief kernel for creating the A matrix as [x^2 xy y^2 x y] that will be used to calculate the covariance matrix (A^T * A)
    * @param A matrix to store the results
    * @param x x coordinates
    * @param y y coordinates
    * @param size number of points
    */
    __global__
    void createAMatrix(float *A, int *x, int *y, int size) {
        // We must launch the kernel with size threads (try with *5 threads)
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {

            float mean_x = fdividef(sum_x, size);
            float mean_y = fdividef(sum_y, size);

            float x_ = x[index] - mean_x;
            float y_ = y[index] - mean_y;

            A[index * 5] = x_ * x_;
            A[index * 5 + 1] = x_ * y_;
            A[index * 5 + 2] = y_ * y_;
            A[index * 5 + 3] = x_;
            A[index * 5 + 4] = y_;
        }
    }

    /**
    * @brief kernel for creating the B matrix as sum of the columns of A
    * @param A matrix to store the results
    * @param B matrix to store the results
    * @param size number of points
    */
    __global__
    void createBMatrix(float *A, float *B, int size) {
        // We must launch 5 blocks of (size) threads (one for each column of A)
        if (threadIdx.x < size) {
            atomicAdd(&B[blockIdx.x], A[threadIdx.x * 5 + blockIdx.x]);
        }
    }

    //------------------------------------------- GET CURVES KERNELS -------------------------------------------------

    /**
    * @brief find the root of a chain
    * @param labels labels array
    * @param label label to find
    * @return root of the chain
    */
    __device__ __inline__ unsigned int find_root(unsigned int *labels, unsigned int label) {
        // Resolve Label
        unsigned int next = labels[label];

        // Follow chain
        while (label != next) {
            // Move to next
            label = next;
            next = labels[label];
        }

        // Return label
        return (label);
    }

    /**
    * @brief reduce the labels of two chains
    * @param g_labels labels array
    * @param label1 first label
    * @param label2 second label
    * @return new label
    */
    __device__ __inline__ unsigned int reduction(unsigned int *g_labels, unsigned int label1, unsigned int label2) {
        // Get next labels
        unsigned int next1 = (label1 != label2) ? g_labels[label1] : 0;
        unsigned int next2 = (label1 != label2) ? g_labels[label2] : 0;

        // Find label1
        while ((label1 != label2) && (label1 != next1)) {
            // Adopt label
            label1 = next1;

            // Fetch next label
            next1 = g_labels[label1];
        }

        // Find label2
        while ((label1 != label2) && (label2 != next2)) {
            // Adopt label
            label2 = next2;

            // Fetch next label
            next2 = g_labels[label2];
        }

        unsigned int label3;
        // While Labels are different
        while (label1 != label2) {
            // Label 2 should be smallest
            if (label1 < label2) {
                // Swap Labels
                label1 = label1 ^ label2;
                label2 = label1 ^ label2;
                label1 = label1 ^ label2;
            }

            // AtomicMin label1 to label2
            label3 = atomicMin(&g_labels[label1], label2);
            label1 = (label1 == label3) ? label2 : label3;
        }

        // Return label1
        return (label1);
    }

    /**
    * @brief kernel for initializing the labels
    * @param g_labels labels array
    * @param g_image image array
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    init_labels(unsigned int *g_labels, const unsigned char *g_image, int numCols, int numRows) {
        // Calculate index
        const unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
        const unsigned int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

        // Check Thread Range
        if ((ix < numCols) && (iy < numRows)) {
            // Fetch five image values
            const unsigned char pyx = g_image[iy * numCols + ix];

            // Neighbour Connections
            const bool nym1x = (iy > 0) ? (pyx == g_image[(iy - 1) * numCols + ix]) : false;
            const bool nyxm1 = (ix > 0) ? (pyx == g_image[(iy) * numCols + ix - 1]) : false;
            const bool nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy - 1) * numCols + ix - 1]) : false;
            const bool nym1xp1 = ((iy > 0) && (ix < numCols - 1)) ? (pyx == g_image[(iy - 1) * numCols + ix + 1]) : false;

            // Label
            unsigned int label;

            // Initialise Label
            // Label will be chosen in the following order:
            // NW > N > NE > E > current position
            label = (nyxm1) ? iy * numCols + ix - 1 : iy * numCols + ix;
            label = (nym1xp1) ? (iy - 1) * numCols + ix + 1 : label;
            label = (nym1x) ? (iy - 1) * numCols + ix : label;
            label = (nym1xm1) ? (iy - 1) * numCols + ix - 1 : label;

            // Write to Global Memory
            g_labels[iy * numCols + ix] = label;
        }
    }

    /**
    * @brief kernel for resolving the labels
    * @param g_labels labels array
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void resolve_labels(unsigned int *g_labels,
                                int numCols, int numRows) {
        // Calculate index
        const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
                                ((blockIdx.x * blockDim.x) + threadIdx.x);

        // Check Thread Range
        if (id < (numRows * numCols)) {
            // Resolve Label
            g_labels[id] = find_root(g_labels, g_labels[id]);
        }
    }

    /**
    * @brief kernel for reducing the labels
    * @param g_labels labels array
    * @param g_image image array
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void label_reduction(unsigned int *g_labels, const unsigned char *g_image,
                                    int numCols, int numRows) {
        // Calculate index
        const unsigned int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
        const unsigned int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);

        // Check Thread Range
        if ((ix < numCols) && (iy < numRows)) {
            // Compare Image Values
            const unsigned char pyx = g_image[iy * numCols + ix];
            const bool nym1x = (iy > 0) ? (pyx == g_image[(iy - 1) * numCols + ix]) : false;

            if (!nym1x) {
                // Neighbouring values
                const bool nym1xm1 = ((iy > 0) && (ix > 0)) ? (pyx == g_image[(iy - 1) * numCols + ix - 1]) : false;
                const bool nyxm1 = (ix > 0) ? (pyx == g_image[(iy) * numCols + ix - 1]) : false;
                const bool nym1xp1 = ((iy > 0) && (ix < numCols - 1)) ? (pyx == g_image[(iy - 1) * numCols + ix + 1])
                                                                    : false;

                if (nym1xp1) {
                    // Check Criticals
                    // There are three cases that need a reduction
                    if ((nym1xm1 && nyxm1) || (nym1xm1 && !nyxm1)) {
                        // Get labels
                        unsigned int label1 = g_labels[(iy) * numCols + ix];
                        unsigned int label2 = g_labels[(iy - 1) * numCols + ix + 1];

                        // Reduction
                        reduction(g_labels, label1, label2);
                    }

                    if (!nym1xm1 && nyxm1) {
                        // Get labels
                        unsigned int label1 = g_labels[(iy) * numCols + ix];
                        unsigned int label2 = g_labels[(iy) * numCols + ix - 1];

                        // Reduction
                        reduction(g_labels, label1, label2);
                    }
                }
            }
        }
    }

    /**
    * @brief kernel for resolving the background
    * @param g_labels labels array
    * @param g_image image array
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void resolve_background(unsigned int *g_labels, const unsigned char *g_image,
                                    int numCols, int numRows) {
        // Calculate index
        const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
                                ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols)
            g_labels[id] = (g_image[id] > 0) ? g_labels[id] + 1 : 0;
    }

    __device__ int translationIndex = 1;

    /**
    * @brief kernel for calculating the translation from sparse form of the labels to dense form
    * @param g_labels labels array
    * @param g_translation translation array
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__
    void calculateTranslation(unsigned int *g_labels, unsigned int *g_translation,
                            int numCols, int numRows) {
        // Calculate index
        const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
                                ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols && g_labels[id] > 0) {
            // translate label
            if (atomicCAS(&g_translation[g_labels[id]], 0, 1) == 0) {
                g_translation[g_labels[id]] = atomicAdd(&translationIndex, 1);
            }
        }
    }

    /**
    * @brief kernel for calculating the centroid of each label
    * @param g_labels labels array
    * @param g_sum_x sum of x coordinates
    * @param g_sum_y sum of y coordinates
    * @param g_total total number of pixels
    * @param g_translation translation array
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__
    void calculateCentroid(unsigned int *g_labels, unsigned int *g_sum_x, unsigned int *g_sum_y, unsigned int *g_total,
                        unsigned int *g_translation, int numCols, int numRows) {

        // Calculate index
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols && g_translation[g_labels[id]] > 0) {
            // translate label
            unsigned int label = g_translation[g_labels[id]];
            atomicAdd(&g_sum_x[label], id % numCols);
            atomicAdd(&g_sum_y[label], id / numCols);
            atomicAdd(&g_total[label], 1);
        }
    }

    /**
    * @brief kernel for calculating if the line is curved enough and his inner gray value
    * @param pic original image
    * @param g_labels labels array with all the labels
    * @param g_sum_x sum of x coordinates of each label
    * @param g_sum_y sum of y coordinates of each label
    * @param g_total total number of pixels of each label
    * @param g_translation translation array from sparse to dense form of labels
    * @param excentricity array of bools that indicates if the line is curved enough
    * @param g_inner_gray array of inner gray values
    * @param mean_dist mean distance between the centroid and the pixels of the label
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    calculateExcentricityInnerGray(unsigned char *pic, unsigned int *g_labels, unsigned int *g_sum_x, unsigned int *g_sum_y,
                                unsigned int *g_total, unsigned int *g_translation, bool *excentricity,
                                unsigned int *g_inner_gray, double mean_dist, int numCols, int numRows) {
        // Calculate index
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols && g_translation[g_labels[id]] > 0) {
            // translate label
            unsigned int label = g_translation[g_labels[id]];
            // calculate valid centroid
            float mean_x = floorf(fdividef(float(g_sum_x[label]), float(g_total[label])) + 0.5);
            float mean_y = floorf(fdividef(float(g_sum_y[label]), float(g_total[label])) + 0.5);
            int x = id % numCols;
            int y = id / numCols;
            if (excentricity[label] && fabsf(mean_x - x) <= mean_dist && fabsf(mean_y - y) <= mean_dist)
                excentricity[label] = false;
            //calc inner mean gray
            if (pic[(numCols * (y + 1)) + (x)] != 0 || pic[(numCols * (y - 1)) + (x)] != 0)
                if (sqrtf(powf(float(y - mean_y), 2) +
                        powf(float(x - mean_x) + 2, 2)) <
                    sqrtf(powf(float(y - mean_y), 2) +
                        powf(float(x - mean_x) - 2, 2)))
                    atomicAdd(&g_inner_gray[label], (unsigned char) pic[(numCols * (y)) + (x) + 2]);
                else
                    atomicAdd(&g_inner_gray[label], (unsigned char) pic[(numCols * (y)) + (x) - 2]);
            else if (pic[(numCols * (y)) + (x + 1)] != 0 || pic[(numCols * (y)) + (x - 1)] != 0)
                if (sqrtf(powf(float(y - mean_y) + 2, 2) +
                        powf(float(x - mean_x), 2)) <
                    sqrtf(powf(float(y - mean_y) - 2, 2) +
                        powf(float(x - mean_x), 2)))
                    atomicAdd(&g_inner_gray[label], (unsigned char) pic[(numCols * (y) + 2) + (x)]);
                else
                    atomicAdd(&g_inner_gray[label], (unsigned char) pic[(numCols * (y) - 2) + (x)]);
            else if (pic[(numCols * (y + 1)) + (x + 1)] != 0 || pic[(numCols * (y - 1)) + (x - 1)] != 0)
                if (sqrtf(powf(float(y - mean_y) - 2, 2) +
                        powf(float(x - mean_x) + 2, 2)) <
                    sqrtf(powf(float(y - mean_y) + 2, 2) +
                        powf(float(x - mean_x) - 2, 2)))
                    atomicAdd(&g_inner_gray[label], (unsigned char) pic[(numCols * (y - 2)) + (x + 2)]);
                else
                    atomicAdd(&g_inner_gray[label], (unsigned char) pic[(numCols * (y + 2)) + (x - 2)]);
            else if (pic[(numCols * (y - 1)) + (x + 1)] != 0 || pic[(numCols * (y + 1)) + (x - 1)] != 0)
                if (sqrtf(powf(float(y - mean_y) + 2, 2) +
                        powf(float(x - mean_x) + 2, 2)) <
                    sqrtf(powf(float(y - mean_y) - 2, 2) +
                        powf(float(x - mean_x) - 2, 2)))
                    atomicAdd(&g_inner_gray[label], (unsigned char) pic[(numCols * (y + 2)) + (x + 2)]);
                else
                    atomicAdd(&g_inner_gray[label], (unsigned char) pic[(numCols * (y - 2)) + (x - 2)]);
        }
    }

    __device__ int bestLabel = 0;

    /**
    * Select the best curve from the given labels
    * @param g_total total number of pixels of each label
    * @param g_translation translation array from sparse to dense form of labels
    * @param excentricity array of bools that indicates if the line is curved enough
    * @param g_inner_gray array of inner gray values
    * @param inner_gray_range range of inner gray values
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__
    void selectBestCurve(unsigned int *g_total, unsigned int *g_translation, bool *excentricity, unsigned int *g_inner_gray,
                        double inner_gray_range, int numCols, int numRows) {
        int bestSize = 0;
        int bestGray = 1000;
        bestLabel = 0;

        for (int i = 1; i < translationIndex; i++) {
            if (excentricity[i]) {
                int inner;
                g_total[i] == 0 ? inner = 0 : inner = (int) floorf(
                        fdividef(float(g_inner_gray[i]), float(g_total[i])) + 0.5);
                if (inner < bestGray - inner_gray_range) {
                    bestGray = inner;
                    bestSize = g_total[i];
                    bestLabel = i;
                } else if (inner < bestGray + inner_gray_range && g_total[i] > bestSize) {
                    bestGray = inner;
                    bestSize = g_total[i];
                    bestLabel = i;
                }
            }
        }
    }

    /**
    * Get the points of the best curve
    * @param g_labels array of labels
    * @param g_translation translation array from sparse to dense form of labels
    * @param out output array
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__
    void getPointsOfBestCurve(unsigned int *g_labels, unsigned int *g_translation, unsigned char *out,
                            int numCols, int numRows) {
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numCols * numRows && g_translation[g_labels[id]] == bestLabel) {
            out[id] = g_translation[g_labels[id]];
        } else if (id < numCols * numRows) {
            out[id] = 0;
        }
    }

    /**
    * @brief kernel for calculating if the line is curved enough (AND NOT INNER GRAY)
    * @param pic original image
    * @param g_labels labels array with all the labels
    * @param g_sum_x sum of x coordinates of each label
    * @param g_sum_y sum of y coordinates of each label
    * @param g_total total number of pixels of each label
    * @param g_translation translation array from sparse to dense form of labels
    * @param excentricity array of bools that indicates if the line is curved enough
    * @param mean_dist mean distance between the centroid and the pixels of the label
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    calculateExcentricity(unsigned char *pic, unsigned int *g_labels, unsigned int *g_sum_x, unsigned int *g_sum_y,
                        unsigned int *g_total, unsigned int *g_translation, bool *excentricity, double mean_dist,
                        int numCols, int numRows) {
        // Calculate index
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols && g_translation[g_labels[id]] > 0) {
            // translate label
            unsigned int label = g_translation[g_labels[id]];
            // calculate valid centroid
            float mean_x = floorf(fdividef(float(g_sum_x[label]), float(g_total[label])) + 0.5);
            float mean_y = floorf(fdividef(float(g_sum_y[label]), float(g_total[label])) + 0.5);
            int x = id % numCols;
            int y = id / numCols;
            if (excentricity[label] && fabsf(mean_x - x) <= mean_dist && fabsf(mean_y - y) <= mean_dist)
                excentricity[label] = false;
        }
    }

    /**
    * @brief kernel for calculating all curves with excentricity
    * @param g_labels map of labels
    * @param excentricity excenricity array
    * @param g_translation translation array from sparse to dense form of labels
    * @param out excentricity map
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__
    void getAllCurves(unsigned int *g_labels, bool *excentricity, unsigned int *g_translation, unsigned char *out,
                    int numCols, int numRows) {
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numCols * numRows && g_translation[g_labels[id]] > 0) {
            out[id] = (g_translation[g_labels[id]]) * excentricity[g_translation[g_labels[id]]];
        } else if (id < numCols * numRows) {
            out[id] = 0;
        }
    }

    //---------------------------------------------- FIND BEST EDGE -----------------------------------------------------------------

    __device__ int atomicInnerGrayIndex = 0;

    /**
    * @brief kernel for getting the points of the best inner gray curve
    * @param edges array of labeled edges
    * @param x array of x coordinates of the points
    * @param y array of y coordinates of the points
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    getPointsOfInnerGrayCurves(unsigned char *edges, int *x, int *y, int numCols, int numRows) {
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numCols * numRows &&
            edges[id] != 0) {
            // The point is part of a true edge
            int j = id / numCols;
            int i = id % numCols;
            // Get atomic index
            int index = atomicAdd(&atomicInnerGrayIndex, 1);
            // Create index for next ellipse fit [x², xy, y², x, y]
            x[index] = i;
            y[index] = j;
        }
    }

    //---------------------------------------------- OPTIMIZE POS KERNEL ------------------------------------------------------------

    __device__ int min_val = 100000;
    __device__ int pos_x = 0;
    __device__ int pos_y = 0;
    __device__ int pos_count = 0;

    /**
    * @brief kernel for calculating min values of the region in order to find the best position
    * @param pic input image
    * @param min_vals array of minimum values
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param reg_size size of the region
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    optimizePos(unsigned char *pic, unsigned int *min_vals, int start_x, int end_x, int start_y, int end_y, int reg_size,
                int numCols, int numRows) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);

        if (j < (end_y - start_y)) {
            i += start_x;
            j += start_y;
            int min_akt = 0;
    #pragma unroll
            for (int k1 = -reg_size; k1 < reg_size; k1++)
    #pragma unroll
                    for (int k2 = -reg_size; k2 < reg_size; k2++) {

                        if (i + k1 > 0 && i + k1 < numCols && j + k2 > 0 && j + k2 < numRows) {
                            int val = ((int) pic[(numCols * j) + (i)] - (int) pic[(numCols * (j + k2)) + (i + k1)]);
                            if (val > 0)
                                min_akt += val;
                        }
                    }
            min_vals[i * numCols + j] = min_akt;
            atomicMin(&min_val, min_akt);
        }
    }

    /**
    * @brief kernel for searching the best position by counting the number of pixels with the minimum value
    * @param min_vals array of minimum values
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__
    void searchPos(unsigned int *min_vals, int start_x, int end_x, int start_y, int end_y,
                int numCols, int numRows) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);

        if (j < (end_y - start_y)) {
            i += start_x;
            j += start_y;
            if (min_vals[i * numCols + j] == min_val) {
                atomicAdd(&pos_count, 1);
                atomicAdd(&pos_x, i);
                atomicAdd(&pos_y, j);
            }
        }
    }

    //---------------------------------------------- ZERO ARROUND TH BORDER KERNELS ------------------------------------------------------------

    /**
    * @brief kernel for calculating the border around the threshold
    * @param pic input image
    * @param edges array of edges
    * @param th_edges array of threshold edges
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param numCols number of columns
    * @param numRows number of rows
    * @param th threshold
    * @param edge_to_th distance from the edge to the threshold
    */
    __global__
    void calculateBorderArroundTh(unsigned char *pic, unsigned char *edges, unsigned char *th_edges, int start_x, int end_x,
                                int start_y, int end_y, int numCols, int numRows, int th, int edge_to_th) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int j = index / (end_x - start_x);
        int i = index % (end_x - start_x);

        if (j < (end_y - start_y)) {
            i += start_x;
            j += start_y;
            if (pic[(numCols * j) + (i)] < th) {
    #pragma unroll
                for (int k1 = -edge_to_th; k1 < edge_to_th; k1++)
    #pragma unroll
                        for (int k2 = -edge_to_th; k2 < edge_to_th; k2++) {
                            if (i + k1 >= 0 && i + k1 < numCols && j + k2 > 0 && j + k2 < numRows &&
                                (int) edges[(numCols * (j + k2)) + (i + k1)]) {
                                th_edges[(numCols * (j + k2)) + (i + k1)] = 255;
                            }
                        }
            }
        }
    }

    /**
    * @brief kernel for calculating the pixels round the previous border
    * @param edges array of threshold edges
    * @param end_x value for the range in x axis (from -end_x to end_x)
    * @param end_y value for the range in y axis (from -end_y to end_y)
    * @param pos_x position in x axis
    * @param pos_y position in y axis
    * @param ret array of the pixels around the border
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    calculateRays(unsigned char *edges, int end_x, int end_y, int pos_x, int pos_y, int *ret, int numCols, int numRows) {
        int ray_idx = blockIdx.x;
        int idx_real = threadIdx.x + 1;

        for (int idx = idx_real; idx < end_x; idx += blockDim.x) {

            int offset_y = _raysPrecalc[0][ray_idx] * idx;
            int offset_x = _raysPrecalc[1][ray_idx] * idx;

            if (((offset_x < end_x && offset_y < end_y)
                && (offset_x > -end_x && offset_y > -end_y))
                && pos_x + offset_x >= 0 && pos_x + offset_x < numCols
                && pos_y + offset_y >= 0 && pos_y + offset_y < numRows
                && (int)edges[(numCols * (pos_y + offset_y)) + (pos_x + offset_x)]
                    != 0) {
                // Save closest edge to the center position
                atomicMin(&ret[ray_idx], idx);
                __syncthreads();
                // If the current edge is the closest, save his label
                if (idx == ret[ray_idx]) {
                        ret[ray_idx] = (int)edges
                            [(numCols * (pos_y + offset_y)) + (pos_x + offset_x)];
                }
            }
        }
    }

    __device__ int atomicIndexRaysPoints = 0;

    /**
    * @brief kernel for calculating the points that are on the curves that hit with the rays
    * @param edges array of labeled edges
    * @param ret array of the pixels around the border
    * @param x array of x coordinates of the points
    * @param y array of y coordinates of the points
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    getPointsOfRaysCurves(unsigned char *edges, int *ret, int *x, int *y, int numCols, int numRows) {
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numCols * numRows &&
            ((int) edges[id] == ret[0] || (int) edges[id] == ret[1] || (int) edges[id] == ret[2] ||
            (int) edges[id] == ret[3] || (int) edges[id] == ret[4] || (int) edges[id] == ret[5] ||
            (int) edges[id] == ret[6] || (int) edges[id] == ret[7])) {
            // The point is part of a true edge
            int j = id / numCols;
            int i = id % numCols;
            // Get atomic index
            int index = atomicAdd(&atomicIndexRaysPoints, 1);
            // Create index for next ellipse fit [x², xy, y², x, y]
            x[index] = i;
            y[index] = j;
        }
    }

    //---------------------------------------------- END OF CUDA KERNELS ------------------------------------------------------------

    /**
    * @brief function for ellipse fitting
    * @param size number of points
    * @return value of the ellipse (a,b,angle,center(x,))
    */
    static cv::RotatedRect GPUfitEllipse(int size) {
        cv::RotatedRect box;
        box.center.x = -1;
        box.center.y = -1;
        box.size.width = -1;
        box.size.height = -1;
        box.angle = -1;

        if (size >= 5) {
            float alpha = 1.0f, beta = 0.0f;
            float cos_phi, sin_phi, mean_x, mean_y, orientation_rad, a, c, d, e;

            //Create A matrix
            //First we must calculate mean of x and y
            cudaMemset(_d_sum_y, 0, sizeof(int));
            cudaMemset(_d_sum_x, 0, sizeof(int));
            sumPoints<<<size / 256 + 1, 256>>>(d_xx, d_yy, size);
            cudaMemset(d_a, 0, 5 * size * sizeof(float));
            createAMatrix<<<size / 256 + 1, 256>>>(d_a, d_xx, d_yy, size);
            cudaMemset(d_b, 0, 5 * sizeof(float));
            cudaMemset(d_b, 0, 5 * sizeof(float));
            createBMatrix<<<5, size>>>(d_a, d_b, size);

            // Calculate covariance matrix (A^t*A)
            cublasSgemm(d_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, size, &alpha, d_a, 5, d_a, 5, &beta, d_a,
                        5);

            // Solving through PSEUDOINVERSE (A^T*A)-1 * A^T * b (with A = A^T*A)
            cublasSgemm(d_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, size, &alpha, d_a, 5, d_a, 5, &beta, d_a2inv,
                        5);
            // Inverse (A^T*A)-1
            cublasSmatinvBatched(d_handle, 5, d_array, 5, d_array, 5, d_info, 1);

            // Multiply (A^T*A)-1 * A^T
            cublasSgemm(d_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, 5, &alpha, d_a2inv, 5, d_a, 5, &beta, d_a,
                        5);
            // Multiply (A^T*A)-1 * A^T
            // I dont know why it is quite fast to do the 5x5 * 5x5 matrix multiplication instead of 5x5 * 5x1 matrix multiplication, so I do it this way and take the first column of the result
            cublasSgemm(d_handle, CUBLAS_OP_N, CUBLAS_OP_N, 5, 5, 5, &alpha, d_a, 5, d_b, 5, &beta, d_x,
                        5);

            //Pass result to host
            cudaMemcpy(h_x, d_x, 5 * sizeof(float), cudaMemcpyDeviceToHost);
            // Pass sum
            int h_sum_x;
            int h_sum_y;

            cudaMemcpy(&h_sum_x, _d_sum_x, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_sum_y, _d_sum_y, sizeof(int), cudaMemcpyDeviceToHost);

            float mean_xx = h_sum_x / float(size);
            float mean_yy = h_sum_y / float(size);

            float a1 = h_x[0];
            float b1 = h_x[1];
            float c1 = h_x[2];
            float d1 = h_x[3];
            float e1 = h_x[4];

            float orientationTolerance = 1e-3;

            if (min(std::abs(b1 / a1), std::abs(b1 / c1)) > orientationTolerance) {
                orientation_rad = 0.5 * atan(b1 / (c1 - a1));
                cos_phi = cos(orientation_rad);
                sin_phi = sin(orientation_rad);
                a = a1 * cos_phi * cos_phi - b1 * cos_phi * sin_phi + c1 * sin_phi * sin_phi;
                c = a1 * sin_phi * sin_phi + b1 * cos_phi * sin_phi + c1 * cos_phi * cos_phi;
                d = d1 * cos_phi - e1 * sin_phi;
                e = d1 * sin_phi + e1 * cos_phi;
                mean_x = cos_phi * mean_xx - sin_phi * mean_yy;
                mean_y = sin_phi * mean_xx + cos_phi * mean_yy;
            } else {
                orientation_rad = 0;
                cos_phi = cos(orientation_rad);
                sin_phi = sin(orientation_rad);
                a = a1;
                c = c1;
                d = d1;
                e = e1;
                mean_x = mean_xx;
                mean_y = mean_yy;
            }

            if (a * c <= 0) {
                return box;
            }

            if (a < 0) {
                a = -a;
                c = -c;
                d = -d;
                e = -e;
            }

            float X0 = mean_x - d / 2 / a;
            float Y0 = mean_y - e / 2 / c;
            float F = 1 + (d * d) / (4 * a) + (e * e) / (4 * c);
            float aa = sqrt(F / a);
            float bb = sqrt(F / c);

            // Apply rotation to center point
            float X0_in = cos_phi * X0 + sin_phi * Y0;
            float Y0_in = -sin_phi * X0 + cos_phi * Y0;

            box.center.x = X0_in;
            box.center.y = Y0_in;
            if (aa > bb) {
                box.size.width = 2 * aa;
                box.size.height = 2 * bb;
                box.angle = -orientation_rad * 180 / float(M_PI) + 90;
            } else {
                box.size.width = 2 * bb;
                box.size.height = 2 * aa;
                box.angle = -orientation_rad * 180 / float(M_PI);
            }
        }
        return box;
    }

    /**
    * @brief function for performing hysteresis thresholding in CPU
    * @param strong strong edges
    * @param weak weak edges
    * @return binary image
    */
    static Mat cbwselect(const Mat &strong, const Mat &weak) {

        int pic_x = strong.cols;
        int pic_y = strong.rows;

        Mat check = Mat::zeros(pic_y, pic_x, CV_8U);

        int lines[MAX_LINE_CPU] = {0};
        int lines_idx = 0;

        int idx = 0;

        for (int i = 1; i < pic_y - 1; i++) {
            for (int j = 1; j < pic_x - 1; j++) {

                if (strong.at<uchar>(idx + j) != 0 && check.at<uchar>(idx + j) == 0) {

                    check.at<uchar>(idx + j) = 255;
                    lines_idx = 1;
                    lines[0] = idx + j;

                    int akt_idx = 0;

                    while (akt_idx < lines_idx && lines_idx < MAX_LINE_CPU) {

                        int akt_pos = lines[akt_idx];

                        if (akt_pos - pic_x - 1 >= 0 && akt_pos + pic_x + 1 < pic_x * pic_y) {
                            for (int k1 = -1; k1 < 2; k1++) {
                                for (int k2 = -1; k2 < 2; k2++) {

                                    if (check.at<uchar>((akt_pos + (k1 * pic_x)) + k2) == 0 &&
                                        weak.at<uchar>((akt_pos + (k1 * pic_x)) + k2) != 0) {
                                        check.at<uchar>((akt_pos + (k1 * pic_x)) + k2) = 255;
                                        // ML 18.11.20: fixed array boundary access for lines array
                                        if (lines_idx < MAX_LINE_CPU) {
                                            lines[lines_idx] = (akt_pos + (k1 * pic_x)) + k2;
                                            lines_idx++;
                                        }
                                    }
                                }
                            }
                        }
                        akt_idx++;
                    }
                }
            }
            idx += pic_x;
        }

        return check;
    }

    /**
    * @brief function for performing CANNY EDGE in GPU
    * @param cols number of columns
    * @param rows number of rows
    */
    static void canny_impl_gpu(int cols, int rows) {
        //GAUSSIAN AND DERIVATIVE GAUSSIAN FOR X COMPUTATION
        cudaMemset(d_resX, 0, rows * cols * sizeof(float));
        cudaMemset(d_auxSmall, 0, rows * cols * sizeof(float));
        gau1Dcol<<<(rows * cols) / 256 + 1, 256>>>(d_smallPic, d_auxSmall, cols, rows);
        deriv1Drow<<<(rows * cols) / 256 + 1, 256>>>(d_auxSmall, d_resX, cols, rows);

        //GAUSSIAN AND DERIVATIVE GAUSSIAN FOR Y COMPUTATION
        cudaMemset(d_resY, 0, rows * cols * sizeof(float));
        cudaMemset(d_auxSmall, 0, rows * cols * sizeof(float));
        gau1Drow<<<(rows * cols) / 256 + 1, 256>>>(d_smallPic, d_auxSmall, cols, rows);
        deriv1Dcol<<<(rows * cols) / 256 + 1, 256>>>(d_auxSmall, d_resY, cols, rows);

        //MAGNITUDE COMPUTATION
        cudaMemset(d_aux2Small, 0, rows * cols * sizeof(float));
        cudaMemset(d_grayHist, 0, 64 * sizeof(unsigned int));
        hypot<<<(rows * cols) / 256 + 1, 256>>>(d_resX, d_resY, d_smallPic, cols, rows);

        //NORMALIZATION 0-256
        int max = 0;
        cublasIsamax_v2(d_handle, rows * cols, d_smallPic, 1, &max);
        float maxValue = 0;
        cudaMemcpy(&maxValue, &d_smallPic[max - 1], sizeof(float), cudaMemcpyDeviceToHost);
        int min = 0;
        cublasIsamin_v2(d_handle, rows * cols, d_smallPic, 1, &min);
        float minValue = 0;
        cudaMemcpy(&minValue, &d_smallPic[min - 1], sizeof(float), cudaMemcpyDeviceToHost);
        normalize<<<(rows * cols) / 256 + 1, 256>>>(d_smallPic, d_smallPic, cols, rows, 0, 1, minValue, maxValue);

        //NORMALIZATION 0-64
        normalize<<<(rows * cols) / 256 + 1, 256>>>(d_smallPic, d_auxSmall, cols, rows, 0, 63, 0, 1);
        cudaMemset(d_grayHist, 0, 256 * sizeof(unsigned int));
        calcHistInt<<<(rows * cols) / 256 + 1, 256>>>(d_auxSmall, d_grayHist, cols, rows, 0, cols, 0, rows);
        operateHist<<<1, 1>>>(d_grayHist, cols, rows);
        cudaMemset(d_strong, 0, rows * cols * sizeof(float));
        cudaMemset(d_weak, 0, rows * cols * sizeof(float));
        nonMaximaSuppresion<<<(rows * cols) / 256 + 1, 256>>>(d_smallPic, d_resX, d_resY, d_strong, d_weak, cols, rows);

        //HYSTERESIS
        cudaMemset(d_exitSmall, 0, rows * cols * sizeof(float));
        hysteresis_pro<<<dim3(rows / 16 + 1, cols / 16 + 1), dim3(16, 16)>>>(d_strong, d_weak, d_exitSmall, cols, rows);

        return;
    }

    /**
    * @brief function for performing CANNY EDGE in CPU
    * @param pic input image
    * @return binary edge  image
    */
    static cv::Mat canny_impl(cv::Mat *pic) {
        int k_sz = 16;

        float gau[16] = {0.000000220358050f, 0.000007297256405f, 0.000146569312970f, 0.001785579770079f, 0.013193749090229f,
                        0.059130281094460f, 0.160732768610747f, 0.265003534507060f, 0.265003534507060f, 0.160732768610747f,
                        0.059130281094460f, 0.013193749090229f, 0.001785579770079f, 0.000146569312970f, 0.000007297256405f,
                        0.000000220358050f};
        float deriv_gau[16] = {-0.000026704586264f, -0.000276122963398f, -0.003355163265098f, -0.024616683775044f,
                            -0.108194751875585f,
                            -0.278368310241814f, -0.388430056419619f, -0.196732206873178f, 0.196732206873178f,
                            0.388430056419619f,
                            0.278368310241814f, 0.108194751875585f, 0.024616683775044f, 0.003355163265098f,
                            0.000276122963398f, 0.000026704586264f};

        cv::Point anchor = cv::Point(-1, -1);
        float delta = 0;
        int ddepth = -1;

        pic->convertTo(*pic, CV_32FC1);

        cv::Mat gau_x = cv::Mat(1, k_sz, CV_32FC1, &gau);
        cv::Mat deriv_gau_x = cv::Mat(1, k_sz, CV_32FC1, &deriv_gau);

        cv::Mat res_x;
        cv::Mat res_y;

        cv::transpose(*pic, *pic);
        filter2D(*pic, res_x, ddepth, gau_x, anchor, delta, cv::BORDER_REPLICATE);
        cv::transpose(*pic, *pic);
        cv::transpose(res_x, res_x);

        filter2D(res_x, res_x, ddepth, deriv_gau_x, anchor, delta, cv::BORDER_REPLICATE);
        filter2D(*pic, res_y, ddepth, gau_x, anchor, delta, cv::BORDER_REPLICATE);

        cv::transpose(res_y, res_y);
        filter2D(res_y, res_y, ddepth, deriv_gau_x, anchor, delta, cv::BORDER_REPLICATE);
        cv::transpose(res_y, res_y);

        cv::Mat res = cv::Mat::zeros(pic->rows, pic->cols, CV_32FC1);

        float *p_res, *p_x, *p_y;

        for (int i = 0; i < res.rows; i++) {
            p_res = res.ptr<float>(i);
            p_x = res_x.ptr<float>(i);
            p_y = res_y.ptr<float>(i);

            for (int j = 0; j < res.cols; j++) {
                //res.at<float>(j, i)= sqrt( (res_x.at<float>(j, i)*res_x.at<float>(j, i)) + (res_y.at<float>(j, i)*res_y.at<float>(j, i)) );
                //res.at<float>(j, i)=robust_pytagoras_after_MOLAR_MORRIS(res_x.at<float>(j, i), res_y.at<float>(j, i));
                //res.at<float>(j, i)=hypot(res_x.at<float>(j, i), res_y.at<float>(j, i));

                //p_res[j]=__ieee754_hypot(p_x[j], p_y[j]);

                p_res[j] = std::hypot(p_x[j], p_y[j]);
            }
        }

        //th selection
        int PercentOfPixelsNotEdges = 0.7 * res.cols * res.rows;
    //    float ThresholdRatio = 0.4f;

        float high_th = 0;
    //    float low_th = 0;

        int h_sz = 64;
        int hist[64];
        for (int i = 0; i < h_sz; i++)
            hist[i] = 0;

        cv::normalize(res, res, 0, 1, cv::NORM_MINMAX, CV_32FC1);
        cv::Mat res_idx = cv::Mat::zeros(pic->rows, pic->cols, CV_8U);
        cv::normalize(res, res_idx, 0, 63, cv::NORM_MINMAX, CV_32S);

        int *p_res_idx = 0;

        for (int i = 0; i < res.rows; i++) {
            p_res_idx = res_idx.ptr<int>(i);
            for (int j = 0; j < res.cols; j++) {
                hist[p_res_idx[j]]++;
            }
        }

        int sum = 0;

        for (int i = 0; i < h_sz; i++) {
            sum += hist[i];
            if (sum > PercentOfPixelsNotEdges) {
                high_th = float(i + 1) / float(h_sz);
                break;
            }
        }

    //    low_th = ThresholdRatio * high_th;

        //non maximum supression + interpolation
        cv::Mat non_ms = cv::Mat::zeros(pic->rows, pic->cols, CV_8U);
        cv::Mat non_ms_hth = cv::Mat::zeros(pic->rows, pic->cols, CV_8U);

        float ix, iy, grad1, grad2, d;
        char *p_non_ms, *p_non_ms_hth;
        float *p_res_t, *p_res_b;

        for (int i = 1; i < res.rows - 1; i++) {
            p_non_ms = non_ms.ptr<char>(i);
            p_non_ms_hth = non_ms_hth.ptr<char>(i);

            p_res = res.ptr<float>(i);
            p_res_t = res.ptr<float>(i - 1);
            p_res_b = res.ptr<float>(i + 1);

            p_x = res_x.ptr<float>(i);
            p_y = res_y.ptr<float>(i);

            for (int j = 1; j < res.cols - 1; j++) {

                iy = p_y[j];
                ix = p_x[j];

                if ((iy <= 0 && ix > -iy) || (iy >= 0 && ix < -iy)) {

                    d = abs(iy / ix);
                    grad1 = (p_res[j + 1] * (1 - d)) + (p_res_t[j + 1] * d);
                    grad2 = (p_res[j - 1] * (1 - d)) + (p_res_b[j - 1] * d);

                    if (p_res[j] >= grad1 && p_res[j] >= grad2) {
                        p_non_ms[j] = (char) 255;

                        if (p_res[j] > high_th)
                            p_non_ms_hth[j] = (char) 255;
                    }
                }

                if ((ix > 0 && -iy >= ix) || (ix < 0 && -iy <= ix)) {
                    d = abs(ix / iy);
                    grad1 = (p_res_t[j] * (1 - d)) + (p_res_t[j + 1] * d);
                    grad2 = (p_res_b[j] * (1 - d)) + (p_res_b[j - 1] * d);

                    if (p_res[j] >= grad1 && p_res[j] >= grad2) {
                        p_non_ms[j] = (char) 255;
                        if (p_res[j] > high_th)
                            p_non_ms_hth[j] = (char) 255;
                    }
                }

                if ((ix <= 0 && ix > iy) || (ix >= 0 && ix < iy)) {
                    d = abs(ix / iy);
                    grad1 = (p_res_t[j] * (1 - d)) + (p_res_t[j - 1] * d);
                    grad2 = (p_res_b[j] * (1 - d)) + (p_res_b[j + 1] * d);

                    if (p_res[j] >= grad1 && p_res[j] >= grad2) {
                        p_non_ms[j] = (char) 255;
                        if (p_res[j] > high_th)
                            p_non_ms_hth[j] = (char) 255;
                    }
                }

                if ((iy < 0 && ix <= iy) || (iy > 0 && ix >= iy)) {
                    d = abs(iy / ix);
                    grad1 = (p_res[j - 1] * (1 - d)) + (p_res_t[j - 1] * d);
                    grad2 = (p_res[j + 1] * (1 - d)) + (p_res_b[j + 1] * d);

                    if (p_res[j] >= grad1 && p_res[j] >= grad2) {
                        p_non_ms[j] = (char) 255;
                        if (p_res[j] > high_th)
                            p_non_ms_hth[j] = (char) 255;
                    }
                }
            }
        }

        ////bw select
    //    cv::Mat res_lin = cv::Mat::zeros(pic->rows, pic->cols, CV_8U);
    //    bwselect(&non_ms_hth, &non_ms, &res_lin);
    //    pic->convertTo(*pic, CV_8U);

        Mat res_lin = cbwselect(non_ms_hth, non_ms);

        return res_lin;
    }

    /**
    * @brief function to detect peek in order to decide execution path in GPU
    * @param cols number of columns
    * @param rows number of rows
    * @param stddev standard deviation of the image
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param peek_detector_factor factor to detect peek
    * @param bright_region_th threshold to detect bright region
    * @return true if peek is detected
    */
    static bool
    peekgpu(int cols, int rows, double *stddev, int start_x, int end_x, int start_y, int end_y, int peek_detector_factor,
            int bright_region_th) {
        int max_gray = 0;
        int max_gray_pos = 0;
        int mean_gray = 0;
        int mean_gray_cnt = 0;

        cudaMemset(d_grayHist, 0, 256 * sizeof(unsigned int));
        cudaMemset(d_meanFeld, 0, (end_x - start_x) * sizeof(float));
        cudaMemset(d_stdFeld, 0, (end_x - start_x) * sizeof(float));

        histAtomic<<<dim3(cols / 16 + 1, rows / 16 + 1), dim3(16, 16), 256 * sizeof(unsigned int), d_stream_1>>>(d_pic, d_grayHist, cols,
                                                                                        rows, start_x, end_x,
                                                                                        start_y, end_y);
        cudaMemcpyAsync(h_grayHist, d_grayHist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost, d_stream_1);

        calcMeanFeld<<<(cols * rows) / 256 + 1, 256, 0, d_stream_2>>>(d_pic, d_meanFeld, cols, rows, start_x, end_x,
                                                                    start_y,
                                                                    end_y);

        calcMeanDiv<<<cols / 64 + 1, 256, 0, d_stream_2>>>(d_meanFeld, end_x - start_x, end_y - start_y);

        calcStdDev<<<(cols * rows) / 256 + 1, 256, 0, d_stream_2>>>(d_pic, d_stdFeld, d_meanFeld, cols, rows, start_x,
                                                                    end_x,
                                                                    start_y, end_y);

        calcStdDevDiv<<<cols / 64 + 1, 64, 0, d_stream_2>>>(d_stdFeld, end_x - start_x, end_y - start_y);

        cudaStreamSynchronize(d_stream_1);

        for (int i = 0; i < 256; i++)
            if (h_grayHist[i] > 0) {

                mean_gray += h_grayHist[i];
                mean_gray_cnt++;

                if (max_gray < h_grayHist[i]) {
                    max_gray = h_grayHist[i];
                    max_gray_pos = i;
                }
            }

        float stddev_tmp = 0;
        cublasSasum(d_handlePeek, end_x - start_x, d_stdFeld, 1, &stddev_tmp);
        *stddev = double(stddev_tmp) / ((end_x - start_x));

        mean_gray = ceil((double) mean_gray / (double) mean_gray_cnt);

        return (max_gray > (mean_gray * peek_detector_factor) && max_gray_pos > bright_region_th);
    }

    /**
    * @brief function to detect peek in order to decide execution path in CPU
    * @param pic input image
    * @param stddev standard deviation of the image
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param peek_detector_factor factor to detect peek
    * @param bright_region_th threshold to detect bright region
    * @return true if peek is detected
    */
    static bool peek(cv::Mat *pic, double *stddev, int start_x, int end_x, int start_y, int end_y, int peek_detector_factor,
                    int bright_region_th) {
        int gray_hist[256];
        int max_gray = 0;
        int max_gray_pos = 0;
        int mean_gray = 0;
        int mean_gray_cnt = 0;

        for (int i = 0; i < 256; i++)
            gray_hist[i] = 0;

        double mean_feld[1000]; //???
        double std_feld[1000];
        for (int i = start_x; i < end_x; i++) {
            mean_feld[i] = 0;
            std_feld[i] = 0;
        }

        for (int i = start_x; i < end_x; i++)
            for (int j = start_y; j < end_y; j++) {
                int idx = (int) pic->data[(pic->cols * j) + i];
                gray_hist[idx]++;
                mean_feld[i] += idx;
            }

        for (int i = start_x; i < end_x; i++)
            mean_feld[i] = (mean_feld[i] / double(end_y - start_y));

        for (int i = start_x; i < end_x; i++)
            for (int j = start_y; j < end_y; j++) {
                int idx = (int) pic->data[(pic->cols * j) + i];
                std_feld[i] += (mean_feld[i] - idx) * (mean_feld[i] - idx);
            }

        for (int i = start_x; i < end_x; i++)
            std_feld[i] = sqrt(std_feld[i] / double(end_y - start_y));

        *stddev = 0;
        for (int i = start_x; i < end_x; i++) {
            *stddev += std_feld[i];
        }

        *stddev = *stddev / ((end_x - start_x));

        for (int i = 0; i < 256; i++)
            if (gray_hist[i] > 0) {

                mean_gray += gray_hist[i];
                mean_gray_cnt++;

                if (max_gray < gray_hist[i]) {
                    max_gray = gray_hist[i];
                    max_gray_pos = i;
                }
            }

        if (mean_gray_cnt < 1)
            mean_gray_cnt = 1;

        mean_gray = ceil((double) mean_gray / (double) mean_gray_cnt);

        return (max_gray > (mean_gray * peek_detector_factor) && max_gray_pos > bright_region_th);
    }

    /**
    * @brief function to remove points with low angle using morphological operations
    * @param cols number of columns
    * @param rows number of rows
    * @param start_xx start column
    * @param end_xx end column
    * @param start_yy start row
    * @param end_yy end row
    * @return true if peek is detected
    */
    static void remove_points_with_low_anglegpu(int cols, int rows, int start_xx, int end_xx, int start_yy, int end_yy) {
        remove_points_with_low_angle_gpu_1<<<dim3(rows / 16 + 1, cols / 16 + 1), dim3(16, 16)>>>(d_edges, start_xx, end_xx,
                                                                                                start_yy, end_yy,
                                                                                                cols, rows);
        cudaMemcpy(d_edgesAux, d_edges, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
        remove_points_with_low_angle_gpu_21<<<(rows * cols) / 256 + 1, 256>>>(d_edges, d_edgesAux, start_xx, end_xx,
                                                                            start_yy, end_yy,
                                                                            cols, rows);
        cudaMemcpy(d_edges, d_edgesAux, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
        remove_points_with_low_angle_gpu_22<<<(rows * cols) / 256 + 1, 256>>>(d_edges, d_edgesAux, start_xx, end_xx,
                                                                            start_yy, end_yy,
                                                                            cols, rows);

        cudaMemcpy(d_edges, d_edgesAux, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);

        remove_points_with_low_angle_gpu_3<<<(rows * cols) / 256 + 1, 256>>>(d_edges, d_edgesAux, start_xx, end_xx,
                                                                            start_yy, end_yy,
                                                                            cols, rows);
        cudaMemcpy(d_edges, d_edgesAux, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
        remove_points_with_low_angle_gpu_4<<<(rows * cols) / 256 + 1, 256>>>(d_edges, d_edgesAux, start_xx, end_xx,
                                                                            start_yy, end_yy,
                                                                            cols, rows);
        cudaMemcpy(d_edges, d_edgesAux, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
    }


    /**
    * @brief remove_points_with_low_angle
    * @param edge edge image
    * @param start_xx start column
    * @param end_xx end column
    * @param start_yy start row
    * @param end_yy end row
    */
    static void remove_points_with_low_angle(cv::Mat *edge, int start_xx, int end_xx, int start_yy, int end_yy) {

        int start_x = start_xx + 5;
        int end_x = end_xx - 5;
        int start_y = start_yy + 5;
        int end_y = end_yy - 5;

        if (start_x < 5)
            start_x = 5;
        if (end_x > edge->cols - 5)
            end_x = edge->cols - 5;
        if (start_y < 5)
            start_y = 5;
        if (end_y > edge->rows - 5)
            end_y = edge->rows - 5;


    //    imshow("zero step", *edge);

        for (int j = start_y; j < end_y; j++)
            for (int i = start_x; i < end_x; i++) {
                if ((int) edge->data[(edge->cols * (j)) + (i)]) {
                    int box[8];

                    box[0] = (int) edge->data[(edge->cols * (j - 1)) + (i - 1)];
                    box[1] = (int) edge->data[(edge->cols * (j - 1)) + (i)];
                    box[2] = (int) edge->data[(edge->cols * (j - 1)) + (i + 1)];
                    box[3] = (int) edge->data[(edge->cols * (j)) + (i + 1)];
                    box[4] = (int) edge->data[(edge->cols * (j + 1)) + (i + 1)];
                    box[5] = (int) edge->data[(edge->cols * (j + 1)) + (i)];
                    box[6] = (int) edge->data[(edge->cols * (j + 1)) + (i - 1)];
                    box[7] = (int) edge->data[(edge->cols * (j)) + (i - 1)];

                    bool valid = false;

                    for (int k = 0; k < 8 && !valid; k++)
                        //if( box[k] && (box[(k+3)%8] || box[(k+4)%8] || box[(k+5)%8]) ) valid=true;
                        if (box[k] && (box[(k + 2) % 8] || box[(k + 3) % 8] || box[(k + 4) % 8] || box[(k + 5) % 8] ||
                                    box[(k + 6) % 8]))
                            valid = true;

                    if (!valid)
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                }
            }

    //    imshow("first step", *edge);

        for (int j = start_y; j < end_y; j++)
            for (int i = start_x; i < end_x; i++) {
                int box[9];

                box[4] = (int) edge->data[(edge->cols * (j)) + (i)];

                if (box[4]) {
                    box[1] = (int) edge->data[(edge->cols * (j - 1)) + (i)];
                    box[3] = (int) edge->data[(edge->cols * (j)) + (i - 1)];
                    box[5] = (int) edge->data[(edge->cols * (j)) + (i + 1)];
                    box[7] = (int) edge->data[(edge->cols * (j + 1)) + (i)];

                    if ((box[5] && box[7]))
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if ((box[5] && box[1]))
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if ((box[3] && box[7]))
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if ((box[3] && box[1]))
                        edge->data[(edge->cols * (j)) + (i)] = 0;

                    //if( (box[1] && box[5]) || (box[1] && box[3]) || (box[3] && box[7]) || (box[5] && box[7]) )
                    //		edge->data[(edge->cols*(j))+(i)]=0;
                }
            }

    //    imshow("second step", *edge);

        for (int j = start_y; j < end_y; j++)
            for (int i = start_x; i < end_x; i++) {
                int box[17];

                box[4] = (int) edge->data[(edge->cols * (j)) + (i)];

                if (box[4]) {
                    box[0] = (int) edge->data[(edge->cols * (j - 1)) + (i - 1)];
                    box[1] = (int) edge->data[(edge->cols * (j - 1)) + (i)];
                    box[2] = (int) edge->data[(edge->cols * (j - 1)) + (i + 1)];

                    box[3] = (int) edge->data[(edge->cols * (j)) + (i - 1)];
                    box[5] = (int) edge->data[(edge->cols * (j)) + (i + 1)];

                    box[6] = (int) edge->data[(edge->cols * (j + 1)) + (i - 1)];
                    box[7] = (int) edge->data[(edge->cols * (j + 1)) + (i)];
                    box[8] = (int) edge->data[(edge->cols * (j + 1)) + (i + 1)];

                    //external
                    box[9] = (int) edge->data[(edge->cols * (j)) + (i + 2)];
                    box[10] = (int) edge->data[(edge->cols * (j + 2)) + (i)];

                    box[11] = (int) edge->data[(edge->cols * (j)) + (i + 3)];
                    box[12] = (int) edge->data[(edge->cols * (j - 1)) + (i + 2)];
                    box[13] = (int) edge->data[(edge->cols * (j + 1)) + (i + 2)];

                    box[14] = (int) edge->data[(edge->cols * (j + 3)) + (i)];
                    box[15] = (int) edge->data[(edge->cols * (j + 2)) + (i - 1)];
                    box[16] = (int) edge->data[(edge->cols * (j + 2)) + (i + 1)];

                    if ((box[10] && !box[7]) && (box[8] || box[6])) {
                        edge->data[(edge->cols * (j + 1)) + (i - 1)] = 0;
                        edge->data[(edge->cols * (j + 1)) + (i + 1)] = 0;
                        edge->data[(edge->cols * (j + 1)) + (i)] = 255;
                    }

                    if ((box[14] && !box[7] && !box[10]) && ((box[8] || box[6]) && (box[16] || box[15]))) {
                        edge->data[(edge->cols * (j + 1)) + (i + 1)] = 0;
                        edge->data[(edge->cols * (j + 1)) + (i - 1)] = 0;
                        edge->data[(edge->cols * (j + 2)) + (i + 1)] = 0;
                        edge->data[(edge->cols * (j + 2)) + (i - 1)] = 0;
                        edge->data[(edge->cols * (j + 1)) + (i)] = 255;
                        edge->data[(edge->cols * (j + 2)) + (i)] = 255;
                    }

                    if ((box[9] && !box[5]) && (box[8] || box[2])) {
                        edge->data[(edge->cols * (j + 1)) + (i + 1)] = 0;
                        edge->data[(edge->cols * (j - 1)) + (i + 1)] = 0;
                        edge->data[(edge->cols * (j)) + (i + 1)] = 255;
                    }

                    if ((box[11] && !box[5] && !box[9]) && ((box[8] || box[2]) && (box[13] || box[12]))) {
                        edge->data[(edge->cols * (j + 1)) + (i + 1)] = 0;
                        edge->data[(edge->cols * (j - 1)) + (i + 1)] = 0;
                        edge->data[(edge->cols * (j + 1)) + (i + 2)] = 0;
                        edge->data[(edge->cols * (j - 1)) + (i + 2)] = 0;
                        edge->data[(edge->cols * (j)) + (i + 1)] = 255;
                        edge->data[(edge->cols * (j)) + (i + 2)] = 255;
                    }
                }
            }

    //    imshow("third step", *edge);

        for (int j = start_y; j < end_y; j++)
            for (int i = start_x; i < end_x; i++) {

                int box[33];

                box[4] = (int) edge->data[(edge->cols * (j)) + (i)];

                if (box[4]) {
                    box[0] = (int) edge->data[(edge->cols * (j - 1)) + (i - 1)];
                    box[1] = (int) edge->data[(edge->cols * (j - 1)) + (i)];
                    box[2] = (int) edge->data[(edge->cols * (j - 1)) + (i + 1)];

                    box[3] = (int) edge->data[(edge->cols * (j)) + (i - 1)];
                    box[5] = (int) edge->data[(edge->cols * (j)) + (i + 1)];

                    box[6] = (int) edge->data[(edge->cols * (j + 1)) + (i - 1)];
                    box[7] = (int) edge->data[(edge->cols * (j + 1)) + (i)];
                    box[8] = (int) edge->data[(edge->cols * (j + 1)) + (i + 1)];

                    box[9] = (int) edge->data[(edge->cols * (j - 1)) + (i + 2)];
                    box[10] = (int) edge->data[(edge->cols * (j - 1)) + (i - 2)];
                    box[11] = (int) edge->data[(edge->cols * (j + 1)) + (i + 2)];
                    box[12] = (int) edge->data[(edge->cols * (j + 1)) + (i - 2)];

                    box[13] = (int) edge->data[(edge->cols * (j - 2)) + (i - 1)];
                    box[14] = (int) edge->data[(edge->cols * (j - 2)) + (i + 1)];
                    box[15] = (int) edge->data[(edge->cols * (j + 2)) + (i - 1)];
                    box[16] = (int) edge->data[(edge->cols * (j + 2)) + (i + 1)];

                    box[17] = (int) edge->data[(edge->cols * (j - 3)) + (i - 1)];
                    box[18] = (int) edge->data[(edge->cols * (j - 3)) + (i + 1)];
                    box[19] = (int) edge->data[(edge->cols * (j + 3)) + (i - 1)];
                    box[20] = (int) edge->data[(edge->cols * (j + 3)) + (i + 1)];

                    box[21] = (int) edge->data[(edge->cols * (j + 1)) + (i + 3)];
                    box[22] = (int) edge->data[(edge->cols * (j + 1)) + (i - 3)];
                    box[23] = (int) edge->data[(edge->cols * (j - 1)) + (i + 3)];
                    box[24] = (int) edge->data[(edge->cols * (j - 1)) + (i - 3)];

                    box[25] = (int) edge->data[(edge->cols * (j - 2)) + (i - 2)];
                    box[26] = (int) edge->data[(edge->cols * (j + 2)) + (i + 2)];
                    box[27] = (int) edge->data[(edge->cols * (j - 2)) + (i + 2)];
                    box[28] = (int) edge->data[(edge->cols * (j + 2)) + (i - 2)];

                    box[29] = (int) edge->data[(edge->cols * (j - 3)) + (i - 3)];
                    box[30] = (int) edge->data[(edge->cols * (j + 3)) + (i + 3)];
                    box[31] = (int) edge->data[(edge->cols * (j - 3)) + (i + 3)];
                    box[32] = (int) edge->data[(edge->cols * (j + 3)) + (i - 3)];

                    if (box[7] && box[2] && box[9])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[7] && box[0] && box[10])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[1] && box[8] && box[11])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[1] && box[6] && box[12])
                        edge->data[(edge->cols * (j)) + (i)] = 0;

                    if (box[0] && box[13] && box[17] && box[8] && box[11] && box[21])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[2] && box[14] && box[18] && box[6] && box[12] && box[22])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[6] && box[15] && box[19] && box[2] && box[9] && box[23])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[8] && box[16] && box[20] && box[0] && box[10] && box[24])
                        edge->data[(edge->cols * (j)) + (i)] = 0;

                    if (box[0] && box[25] && box[29] && box[2] && box[27] && box[31])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[0] && box[25] && box[29] && box[6] && box[28] && box[32])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[8] && box[26] && box[30] && box[2] && box[27] && box[31])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[8] && box[26] && box[30] && box[6] && box[28] && box[32])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                }
            }
    //    imshow("last step", *edge);
    }

    /**
    * @brief function to get the curves from the edge image and the most eliptic and dark inside curve in GPU
    * @param cols the number of columns
    * @param rows the number of rows
    * @param start_x the start column
    * @param end_x the end column
    * @param start_y the start row
    * @param end_y the end row
    * @param mean_dist the mean distance between the centroid and the curve
    * @param inner_color_range the range of the inner color
    **/
    static void
    get_curvesgpu(int cols, int rows, int start_x, int end_x, int start_y, int end_y, double mean_dist,
                int inner_color_range) {
        if (start_x < 2)
            start_x = 2;
        if (start_y < 2)
            start_y = 2;
        if (end_x > cols - 2)
            end_x = cols - 2;
        if (end_y > rows - 2)
            end_y = rows - 2;

        cv::Point mean_p;

        // Create Grid/Block
        dim3 block(32, 4);
        dim3 grid((cols + 32 - 1) / 32,
                (rows + 4 - 1) / 4);
        cudaMemset(d_outputImg, 0, cols * rows * sizeof(unsigned int));
        init_labels<<< grid, block >>>(d_outputImg, d_edges, cols, rows);
        resolve_labels <<< grid, block >>>(d_outputImg, cols, rows);
        label_reduction <<< grid, block >>>(d_outputImg, d_edges, cols, rows);
        resolve_labels <<< grid, block >>>(d_outputImg, cols, rows);
        resolve_background<<<grid, block>>>(d_outputImg, d_edges, cols, rows);
        cudaMemset(d_translation, 0, cols * rows * sizeof(unsigned int));
        unsigned int uno = 1;
        cudaMemcpy(_translationIndex, &uno, sizeof(unsigned int), cudaMemcpyHostToDevice);
        calculateTranslation<<<grid, block>>>(d_outputImg, d_translation, cols, rows);
        cudaMemset(d_sum_x, 0, MAX_CURVES * sizeof(unsigned int));
        cudaMemset(d_sum_y, 0, MAX_CURVES * sizeof(unsigned int));
        cudaMemset(d_total, 0, MAX_CURVES * sizeof(unsigned int));
        calculateCentroid<<<grid, block>>>(d_outputImg, d_sum_x, d_sum_y, d_total, d_translation, cols, rows);

        cudaMemset(d_excentricity, true, MAX_CURVES * sizeof(bool));
        cudaMemset(d_innerGray, 0, MAX_CURVES * sizeof(unsigned int));
        if (inner_color_range == 0) {
            //calculate just excentricity
            calculateExcentricity<<<grid, block>>>(d_pic, d_outputImg, d_sum_x, d_sum_y, d_total, d_translation,
                                                d_excentricity, mean_dist, cols, rows);
            getAllCurves<<<grid, block>>>(d_outputImg, d_excentricity, d_translation, d_edges, cols, rows);
        } else {
            // calculate excentricity and inner gray
            calculateExcentricityInnerGray<<<grid, block>>>(d_pic, d_outputImg, d_sum_x, d_sum_y, d_total, d_translation,
                                                            d_excentricity, d_innerGray, mean_dist, cols, rows);
            selectBestCurve<<<1, 1>>>(d_total, d_translation, d_excentricity, d_innerGray, inner_color_range, cols,
                                    rows);
            getPointsOfBestCurve<<<grid, block>>>(d_outputImg, d_translation, d_edges, cols, rows);
        }
    }

    /**
    * @brief function to get the curves from the edge image and the most eliptic and dark inside curve in CPU
    * @param pic the original image
    * @param edge the edge image
    * @param start_x the start column
    * @param end_x the end column
    * @param start_y the start row
    * @param end_y the end row
    * @param mean_dist the mean distance between the centroid and the curve
    * @param inner_color_range the range of the inner color
    * @return a vector of the best curve
    * */
    static std::vector <std::vector<cv::Point>>
    get_curves(cv::Mat *pic, cv::Mat *edge, int start_x, int end_x, int start_y, int end_y, double mean_dist,
            int inner_color_range) {

        std::vector <std::vector<cv::Point>> all_curves;
        std::vector <cv::Point> curve;

        if (start_x < 2)
            start_x = 2;
        if (start_y < 2)
            start_y = 2;
        if (end_x > pic->cols - 2)
            end_x = pic->cols - 2;
        if (end_y > pic->rows - 2)
            end_y = pic->rows - 2;

        int curve_idx = 0;
        cv::Point mean_p;
        bool add_curve;
        int mean_inner_gray;
        int mean_inner_gray_last = 1000000;

        //curve.reserve(1000);
        //all_curves.reserve(1000);

        all_curves.clear();

        bool check[IMG_SIZE][IMG_SIZE];

        for (int i = 0; i < IMG_SIZE; i++)
            for (int j = 0; j < IMG_SIZE; j++)
                check[i][j] = 0;

        for (int i = start_x; i < end_x; i++)
            for (int j = start_y; j < end_y; j++) {

                if (edge->data[(edge->cols * (j)) + (i)] == 255 && !check[i][j]) {
                    check[i][j] = 1;

                    curve.clear();
                    curve_idx = 0;

                    curve.push_back(cv::Point(i, j));
                    mean_p.x = i;
                    mean_p.y = j;
                    curve_idx++;

                    int akt_idx = 0;

                    while (akt_idx < curve_idx) {

                        cv::Point akt_pos = curve[akt_idx];
                        for (int k1 = -1; k1 < 2; k1++)
                            for (int k2 = -1; k2 < 2; k2++) {

                                if (akt_pos.x + k1 >= start_x && akt_pos.x + k1 < end_x && akt_pos.y + k2 >= start_y &&
                                    akt_pos.y + k2 < end_y)
                                    if (!check[akt_pos.x + k1][akt_pos.y + k2])
                                        if (edge->data[(edge->cols * (akt_pos.y + k2)) + (akt_pos.x + k1)] == 255) {
                                            check[akt_pos.x + k1][akt_pos.y + k2] = 1;

                                            mean_p.x += akt_pos.x + k1;
                                            mean_p.y += akt_pos.y + k2;
                                            curve.push_back(cv::Point(akt_pos.x + k1, akt_pos.y + k2));
                                            curve_idx++;
                                        }
                            }
                        akt_idx++;
                    }

                    if (curve_idx > 0 && curve.size() > 0) {
                        add_curve = true;
                        mean_p.x = floor((double(mean_p.x) / double(curve_idx)) + 0.5);
                        mean_p.y = floor((double(mean_p.y) / double(curve_idx)) + 0.5);

                        for (int i = 0; i < curve.size(); i++)
                            if (abs(mean_p.x - curve[i].x) <= mean_dist && abs(mean_p.y - curve[i].y) <= mean_dist)
                                add_curve = false;

    //                    //is ellipse fit possible
    //                    if (add_curve) {
    //                        cv::RotatedRect ellipse = cv::fitEllipse(cv::Mat(curve));
    //
    //                        if (ellipse.center.x < 0 || ellipse.center.y < 0 ||
    //                            ellipse.center.x > pic->cols || ellipse.center.y > pic->rows) {
    //
    //                            add_curve = false;
    //                        }
    //
    //                        if (ellipse.size.height > 2.0 * ellipse.size.width ||
    //                            ellipse.size.width > 2.0 * ellipse.size.height) {
    //
    //                            add_curve = false;
    //                        }
    //                    }

                        if (add_curve) {
                            if (inner_color_range > 0) {
                                mean_inner_gray = 0;

                                //calc inner mean
                                for (int i = 0; i < curve.size(); i++) {

                                    if (pic->data[(pic->cols * (curve[i].y + 1)) + (curve[i].x)] != 0 ||
                                        pic->data[(pic->cols * (curve[i].y - 1)) + (curve[i].x)] != 0)
                                        if (sqrt(pow(double(curve[i].y - mean_p.y), 2) +
                                                pow(double(curve[i].x - mean_p.x) + 2, 2)) <
                                            sqrt(pow(double(curve[i].y - mean_p.y), 2) +
                                                pow(double(curve[i].x - mean_p.x) - 2, 2)))

                                            mean_inner_gray += (unsigned char) pic->data[(pic->cols * (curve[i].y)) +
                                                                                        (curve[i].x + 2)];
                                        else
                                            mean_inner_gray += (unsigned char) pic->data[(pic->cols * (curve[i].y)) +
                                                                                        (curve[i].x - 2)];

                                    else if (pic->data[(pic->cols * (curve[i].y)) + (curve[i].x + 1)] != 0 ||
                                            pic->data[(pic->cols * (curve[i].y)) + (curve[i].x - 1)] != 0)
                                        if (sqrt(pow(double(curve[i].y - mean_p.y + 2), 2) +
                                                pow(double(curve[i].x - mean_p.x), 2)) <
                                            sqrt(pow(double(curve[i].y - mean_p.y - 2), 2) +
                                                pow(double(curve[i].x - mean_p.x), 2)))

                                            mean_inner_gray += (unsigned char) pic->data[(pic->cols * (curve[i].y + 2)) +
                                                                                        (curve[i].x)];
                                        else
                                            mean_inner_gray += (unsigned char) pic->data[(pic->cols * (curve[i].y - 2)) +
                                                                                        (curve[i].x)];

                                    else if (pic->data[(pic->cols * (curve[i].y + 1)) + (curve[i].x + 1)] != 0 ||
                                            pic->data[(pic->cols * (curve[i].y - 1)) + (curve[i].x - 1)] != 0)
                                        if (sqrt(pow(double(curve[i].y - mean_p.y - 2), 2) +
                                                pow(double(curve[i].x - mean_p.x + 2), 2)) <
                                            sqrt(pow(double(curve[i].y - mean_p.y + 2), 2) +
                                                pow(double(curve[i].x - mean_p.x - 2), 2)))

                                            mean_inner_gray += (unsigned char) pic->data[(pic->cols * (curve[i].y - 2)) +
                                                                                        (curve[i].x + 2)];
                                        else
                                            mean_inner_gray += (unsigned char) pic->data[(pic->cols * (curve[i].y + 2)) +
                                                                                        (curve[i].x - 2)];

                                    else if (pic->data[(pic->cols * (curve[i].y - 1)) + (curve[i].x + 1)] != 0 ||
                                            pic->data[(pic->cols * (curve[i].y + 1)) + (curve[i].x - 1)] != 0)
                                        if (sqrt(pow(double(curve[i].y - mean_p.y + 2), 2) +
                                                pow(double(curve[i].x - mean_p.x + 2), 2)) <
                                            sqrt(pow(double(curve[i].y - mean_p.y - 2), 2) +
                                                pow(double(curve[i].x - mean_p.x - 2), 2)))

                                            mean_inner_gray += (unsigned char) pic->data[(pic->cols * (curve[i].y + 2)) +
                                                                                        (curve[i].x + 2)];
                                        else
                                            mean_inner_gray += (unsigned char) pic->data[(pic->cols * (curve[i].y - 2)) +
                                                                                        (curve[i].x - 2)];

                                    //mean_inner_gray+=pic->data[( pic->cols*( curve[i].y+((mean_p.y-curve[i].y)/2) ) ) + ( curve[i].x+((mean_p.x-curve[i].x)/2) )];
                                }

                                mean_inner_gray = floor((double(mean_inner_gray) / double(curve.size())) + 0.5);

                                if (mean_inner_gray_last > (mean_inner_gray + inner_color_range)) {
                                    mean_inner_gray_last = mean_inner_gray;
                                    all_curves.clear();
                                    all_curves.push_back(curve);
                                } else if (mean_inner_gray_last <= (mean_inner_gray + inner_color_range) &&
                                        mean_inner_gray_last >= (mean_inner_gray - inner_color_range)) {
                                    if (curve.size() > all_curves[0].size()) {
                                        mean_inner_gray_last = mean_inner_gray;
                                        all_curves.clear();
                                        all_curves.push_back(curve);
                                    }
                                }
                            } else
                                all_curves.push_back(curve);
                        }
                    }
                }
            }

        /*
        std::cout<<all_curves.size()<<std::endl;
        for(int i=0;i<1;i++)
            for(int j=0;j<all_curves[i].size();j++)
            std::cout<<all_curves[i][j].x<<";"<<all_curves[i][j].y<<std::endl;
        cv::Mat m = cv::Mat::zeros(edge->rows, edge->cols, CV_8U);
        for(int i=0;i<all_curves.size();i++)
            for(int j=0;j<all_curves[i].size();j++)
                m.data[(edge->cols*all_curves[i][j].y)+(all_curves[i][j].x)]=255;
        imshow("ddd",m);
        */

        return all_curves;
    }

    /**
    * @brief function for finding the best edge in CPU
    * @param pic original picture
    * @param edge edge picture
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param mean_dist mean distance between points
    * @param inner_color_range range of inner color
    * @return best edge
    */
    static cv::RotatedRect
    find_best_edge(cv::Mat *pic, cv::Mat *edge, int start_x, int end_x, int start_y, int end_y, double mean_dist,
                int inner_color_range) {
        cv::RotatedRect ellipse;
        ellipse.center.x = 0;
        ellipse.center.y = 0;
        ellipse.angle = 0.0;
        ellipse.size.height = 0.0;
        ellipse.size.width = 0.0;

        std::vector <std::vector<cv::Point>> all_curves = get_curves(pic, edge, start_x, end_x, start_y, end_y, mean_dist,
                                                                    inner_color_range);

        if (all_curves.size() == 1) {
            ellipse = cv::fitEllipse(cv::Mat(all_curves[0]));

            if (ellipse.center.x < 0 || ellipse.center.y < 0 || ellipse.center.x > pic->cols ||
                ellipse.center.y > pic->rows) {
                ellipse.center.x = 0;
                ellipse.center.y = 0;
                ellipse.angle = 0.0;
                ellipse.size.height = 0.0;
                ellipse.size.width = 0.0;
            }
        } else {
            ellipse.center.x = 0;
            ellipse.center.y = 0;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;
        }
        return ellipse;
    }

    /**
    * @brief function for finding the best edge in GPU
    * @param cols number of columns
    * @param rows number of rows
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param mean_dist mean distance between points
    * @param inner_color_range range of inner color
    * @return best edge
    */
    static cv::RotatedRect
    find_best_edgegpu(int cols, int rows, int start_x, int end_x, int start_y, int end_y, double mean_dist,
                    int inner_color_range) {
        cv::RotatedRect ellipse;
        ellipse.center.x = 0;
        ellipse.center.y = 0;
        ellipse.angle = 0.0;
        ellipse.size.height = 0.0;
        ellipse.size.width = 0.0;

        get_curvesgpu(cols, rows, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);

        cudaMemset(_atomicInnerGrayIndex, 0, sizeof(int));
        cudaMemset(d_xx, 0, sizeof(int) * cols * rows);
        cudaMemset(d_yy, 0, sizeof(int) * cols * rows);
        getPointsOfInnerGrayCurves<<<(cols * rows) / 128 + 1, 128>>>(d_edges, d_xx, d_yy, cols,
                                                                    rows);

        int curveSize;
        cudaMemcpy(&curveSize, _atomicInnerGrayIndex, sizeof(int), cudaMemcpyDeviceToHost);

        if (curveSize > 4) {
            ellipse = GPUfitEllipse(curveSize);

            //draw ellipse
            //cv::ellipse(*pic, ellipse, cv::Scalar(255, 0, 0), 2, 8);
            //show image
            //cv::imshow("ellipse", *pic);

            if (ellipse.center.x < 0 || ellipse.center.y < 0 || ellipse.center.x > cols ||
                ellipse.center.y > rows) {
                ellipse.center.x = 0;
                ellipse.center.y = 0;
                ellipse.angle = 0.0;
                ellipse.size.height = 0.0;
                ellipse.size.width = 0.0;
            }
        } else {
            ellipse.center.x = 0;
            ellipse.center.y = 0;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;
        }
        return ellipse;
    }

    /**
    * @brief function for finding the best coarsed position in the angular histograms
    * @param hist angular histogram
    * @param mini minimum value
    * @param max_region_hole maximum hole size
    * @param min_region_size minimum region size
    * @param real_hist_sz real size of the histogram
    * @return best position
    */
    static int calc_pos(int *hist, int mini, int max_region_hole, int min_region_size, int real_hist_sz) {
        int pos = 0;

        int mean_pos = 0;
        int pos_hole = 0;
        int count = 0;
        int hole_size = 0;
        bool region_start = false;

        for (int i = 0; i < DEF_SIZE; i++) {
            if (hist[i] > mini && !region_start) {
                region_start = true;
                count++;
                mean_pos += i;
            } else if (hist[i] > mini && region_start) {
                count += 1 + hole_size;
                mean_pos += i + pos_hole;
                hole_size = 0;
                pos_hole = 0;
            } else if (hist[i] <= mini && region_start && hole_size < max_region_hole) {
                hole_size++;
                pos_hole += i;
            } else if (hist[i] <= mini && region_start && hole_size >= max_region_hole && count >= min_region_size) {

                if (count < 1)
                    count = 1;
                mean_pos = mean_pos / count;
                if (pow(double((real_hist_sz / 2) - mean_pos), 2) < pow(double((real_hist_sz / 2) - pos), 2))
                    pos = mean_pos;

                pos_hole = 0;
                hole_size = 0;
                region_start = 0;
                count = 0;
                mean_pos = 0;
            } else if (hist[i] <= mini && region_start && hole_size >= max_region_hole && count < min_region_size) {
                pos_hole = 0;
                hole_size = 0;
                region_start = 0;
                count = 0;
                mean_pos = 0;
            }
        }

        return pos;
    }

    /**
    * @brief function for calculating the coarsed position in the angular histograms in GPU
    * @param cols number of columns
    * @param rows number of rows
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param th threshold
    * @param th_histo threshold for the angular histogram
    * @param max_region_hole maximum hole size
    * @param min_region_size minimum region size
    * @return best position
    */
    static cv::Point
    th_angular_histogpu(int cols, int rows, int start_x, int end_x, int start_y, int end_y, int th,
                        double th_histo, int max_region_hole, int min_region_size) {
        cv::Point pos(0, 0);
        if (start_x < 0)
            start_x = 0;
        if (start_y < 0)
            start_y = 0;
        if (end_x > cols)
            end_x = cols;
        if (end_y > rows)
            end_y = rows;

        int max_l = 0;
        int max_lb = 0;
        int max_b = 0;
        int max_br = 0;

        int min_l, min_lb, min_b, min_br;
        int pos_l, pos_lb, pos_b, pos_br;

        cudaMemset(d_hist_l, 0, DEF_SIZE * sizeof(int));
        cudaMemset(d_hist_lb, 0, DEF_SIZE * sizeof(int));
        cudaMemset(d_hist_b, 0, DEF_SIZE * sizeof(int));
        cudaMemset(d_hist_br, 0, DEF_SIZE * sizeof(int));

        calculateRotatedHists<<<(cols * rows) / 128 + 1, 128>>>(d_pic, d_hist_l, d_hist_b, d_hist_lb, d_hist_br,
                                                                cols, rows, start_x, end_x, start_y, end_y, th);

        cudaMemcpy(h_hist_l, d_hist_l, DEF_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_hist_lb, d_hist_lb, DEF_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_hist_b, d_hist_b, DEF_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_hist_br, d_hist_br, DEF_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < DEF_SIZE; i++) {
            if (h_hist_l[i] > max_l)
                max_l = h_hist_l[i];
            if (h_hist_lb[i] > max_lb)
                max_lb = h_hist_lb[i];
            if (h_hist_b[i] > max_b)
                max_b = h_hist_b[i];
            if (h_hist_br[i] > max_br)
                max_br = h_hist_br[i];
        }

        min_l = max_l - floor(max_l * th_histo);
        min_lb = max_lb - floor(max_lb * th_histo);
        min_b = max_b - floor(max_b * th_histo);
        min_br = max_br - floor(max_br * th_histo);

        pos_l = calc_pos(h_hist_l, min_l, max_region_hole, min_region_size, rows);
        pos_lb = calc_pos(h_hist_lb, min_lb, max_region_hole, min_region_size, cols + rows);
        pos_b = calc_pos(h_hist_b, min_b, max_region_hole, min_region_size, cols);
        pos_br = calc_pos(h_hist_br, min_br, max_region_hole, min_region_size, cols + rows);

        /*
        std::cout<<"min_l: "<<min_l<<" min_lb: "<<min_lb<<std::endl;
        std::cout<<"min_b: "<<min_b<<" min_br: "<<min_br<<std::endl;
        std::cout<<"l: "<<pos_l<<"    lb: "<<pos_lb<<std::endl;
        std::cout<<"b: "<<pos_b<<"    br: "<<pos_br<<std::endl;*/


        if (pos_l > 0 && pos_lb > 0 && pos_b > 0 && pos_br > 0) {
            pos.x = floor(((pos_b + (floor((((pos_br + rows) - pos_lb) / 2) + 0.5) + pos_lb - rows)) / 2) + 0.5);
            pos.y = floor(((pos_l + (rows - floor((((pos_br + rows) - pos_lb) / 2) + 0.5))) / 2) + 0.5);
        } else if (pos_l > 0 && pos_b > 0) {
            pos.x = pos_b;
            pos.y = pos_l;
        } else if (pos_lb > 0 && pos_br > 0) {
            pos.x = floor((((pos_br + rows) - pos_lb) / 2) + 0.5) + pos_lb - rows;
            pos.y = rows - floor((((pos_br + rows) - pos_lb) / 2) + 0.5);
        }

        if (pos.x < 0)
            pos.x = 0;
        if (pos.y < 0)
            pos.y = 0;
        if (pos.x >= cols)
            pos.x = 0;
        if (pos.y >= rows)
            pos.y = 0;
        return pos;
    }

    /**
    * @brief function for calculating the coarsed position in the angular histograms in CPU
    * @param pic original picture
    * @param pic_th thresholded picture
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param th threshold
    * @param th_histo threshold for the angular histogram
    * @param max_region_hole maximum hole size
    * @param min_region_size minimum region size
    * @return best position
    */
    static cv::Point
    th_angular_histo(cv::Mat *pic, cv::Mat *pic_th, int start_x, int end_x, int start_y, int end_y, int th, double th_histo,
                    int max_region_hole, int min_region_size) {
        cv::Point pos(0, 0);
        if (start_x < 0)
            start_x = 0;
        if (start_y < 0)
            start_y = 0;
        if (end_x > pic->cols)
            end_x = pic->cols;
        if (end_y > pic->rows)
            end_y = pic->rows;

        int max_l = 0;
        int max_lb = 0;
        int max_b = 0;
        int max_br = 0;

        int min_l, min_lb, min_b, min_br;
        int pos_l, pos_lb, pos_b, pos_br;

        int hist_l[DEF_SIZE];
        int hist_lb[DEF_SIZE];
        int hist_b[DEF_SIZE];
        int hist_br[DEF_SIZE];

        for (int i = 0; i < DEF_SIZE; i++) {
            hist_l[i] = 0;
            hist_lb[i] = 0;
            hist_b[i] = 0;
            hist_br[i] = 0;
        }

        int idx_lb = 0;
        int idx_br = 0;

        for (int i = start_x; i < end_x; i++) {
            for (int j = start_y; j < end_y; j++) {

                if (pic->data[(pic->cols * j) + i] < th) {

                    pic_th->data[(pic->cols * j) + i] = 255;

                    idx_lb = (pic->cols / 2) + (i - (pic->cols / 2)) + (j);
                    idx_br = (pic->cols / 2) + (i - (pic->cols / 2)) + (pic->rows - j);

                    if (j >= 0 && j < DEF_SIZE && i >= 0 && i < DEF_SIZE && idx_lb >= 0 && idx_lb < DEF_SIZE &&
                        idx_br >= 0 && idx_br < DEF_SIZE) {

                        if (++hist_l[j] > max_l)
                            max_l = hist_l[j];

                        if (++hist_b[i] > max_b)
                            max_b = hist_b[i];

                        if (++hist_lb[idx_lb] > max_lb)
                            max_lb = hist_lb[idx_lb];

                        if (++hist_br[idx_br] > max_br)
                            max_br = hist_br[idx_br];
                    }
                }
            }
        }

        min_l = max_l - floor(max_l * th_histo);
        min_lb = max_lb - floor(max_lb * th_histo);
        min_b = max_b - floor(max_b * th_histo);
        min_br = max_br - floor(max_br * th_histo);

        pos_l = calc_pos(hist_l, min_l, max_region_hole, min_region_size, pic->rows);
        pos_lb = calc_pos(hist_lb, min_lb, max_region_hole, min_region_size, pic->cols + pic->rows);
        pos_b = calc_pos(hist_b, min_b, max_region_hole, min_region_size, pic->cols);
        pos_br = calc_pos(hist_br, min_br, max_region_hole, min_region_size, pic->cols + pic->rows);


        /*std::cout<<"min_l: "<<min_l<<" min_lb: "<<min_lb<<std::endl;
        std::cout<<"min_b: "<<min_b<<" min_br: "<<min_br<<std::endl;
        std::cout<<"l: "<<pos_l<<"    lb: "<<pos_lb<<std::endl;
        std::cout<<"b: "<<pos_b<<"    br: "<<pos_br<<std::endl;*/


        if (pos_l > 0 && pos_lb > 0 && pos_b > 0 && pos_br > 0) {
            pos.x = floor(((pos_b + (floor((((pos_br + pic->rows) - pos_lb) / 2) + 0.5) + pos_lb - pic->rows)) / 2) + 0.5);
            pos.y = floor(((pos_l + (pic->rows - floor((((pos_br + pic->rows) - pos_lb) / 2) + 0.5))) / 2) + 0.5);
        } else if (pos_l > 0 && pos_b > 0) {
            pos.x = pos_b;
            pos.y = pos_l;
        } else if (pos_lb > 0 && pos_br > 0) {
            pos.x = floor((((pos_br + pic->rows) - pos_lb) / 2) + 0.5) + pos_lb - pic->rows;
            pos.y = pic->rows - floor((((pos_br + pic->rows) - pos_lb) / 2) + 0.5);
        }

        if (pos.x < 0)
            pos.x = 0;
        if (pos.y < 0)
            pos.y = 0;
        if (pos.x >= pic->cols)
            pos.x = 0;
        if (pos.y >= pic->rows)
            pos.y = 0;

        //imshow("th", *pic_th);
    //    cv::ellipse(*pic, cv::RotatedRect(pos, cv::Size2f(5, 5), 0), CV_RGB(255, 255, 255));
        //imshow("angular", *pic);
        //waitKey(0);

        return pos;
    }

    /**
    * @brief grow_region from coarsed position to find the center of the region
    * @param ellipse the ellipse to grow
    * @param pic the picture
    * @param max_ellipse_radi the maximum radius of the ellipse
    */
    static void grow_region(cv::RotatedRect *ellipse, cv::Mat *pic, int max_ellipse_radi) {

        float mean = 0.0;

        int x0 = ellipse->center.x;
        int y0 = ellipse->center.y;

        int mini = 1000;
        int maxi = 0;

        for (int i = -2; i < 3; i++)
            for (int j = -2; j < 3; j++) {
                if (y0 + j > 0 && y0 + j < pic->rows && x0 + i > 0 && x0 + i < pic->cols) {
                    if (mini > pic->data[(pic->cols * (y0 + j)) + (x0 + i)])
                        mini = pic->data[(pic->cols * (y0 + j)) + (x0 + i)];

                    if (maxi < pic->data[(pic->cols * (y0 + j)) + (x0 + i)])
                        maxi = pic->data[(pic->cols * (y0 + j)) + (x0 + i)];

                    mean += pic->data[(pic->cols * (y0 + j)) + (x0 + i)];
                }
            }

        mean = mean / 25.0;

        float diff = abs(mean - pic->data[(pic->cols * (y0)) + (x0)]);

        int th_up = ceil(mean + diff) + 1;
        int th_down = floor(mean - diff) - 1;

        int radi = 0;

        for (int i = 1; i < max_ellipse_radi; i++) {
            radi = i;

            int left = 0;
            int right = 0;
            int top = 0;
            int bottom = 0;

            for (int j = -i; j <= 1 + (i * 2); j++) {

                //left
                if (y0 + j > 0 && y0 + j < pic->rows && x0 + i > 0 && x0 + i < pic->cols)
                    if (pic->data[(pic->cols * (y0 + j)) + (x0 + i)] > th_down &&
                        pic->data[(pic->cols * (y0 + j)) + (x0 + i)] < th_up) {
                        left++;
                        //pic->data[(pic->cols*(y0+j))+(x0+i)]=255;
                    }

                //right
                if (y0 + j > 0 && y0 + j < pic->rows && x0 - i > 0 && x0 - i < pic->cols)
                    if (pic->data[(pic->cols * (y0 + j)) + (x0 - i)] > th_down &&
                        pic->data[(pic->cols * (y0 + j)) + (x0 - i)] < th_up) {
                        right++;
                        //pic->data[(pic->cols*(y0+j))+(x0-i)]=255;
                    }

                //top
                if (y0 - i > 0 && y0 - i < pic->rows && x0 + j > 0 && x0 + j < pic->cols)
                    if (pic->data[(pic->cols * (y0 - i)) + (x0 + j)] > th_down &&
                        pic->data[(pic->cols * (y0 - i)) + (x0 + j)] < th_up) {
                        top++;
                        //pic->data[(pic->cols*(y0-i))+(x0+j)]=255;
                    }

                //bottom
                if (y0 + i > 0 && y0 + i < pic->rows && x0 + j > 0 && x0 + j < pic->cols)
                    if (pic->data[(pic->cols * (y0 + i)) + (x0 + j)] > th_down &&
                        pic->data[(pic->cols * (y0 + i)) + (x0 + j)] < th_up) {
                        bottom++;
                        //pic->data[(pic->cols*(y0+i))+(x0+j)]=255;
                    }
            }

            //if less than 25% stop
            float p_left = float(left) / float(1 + (2 * i));
            float p_right = float(right) / float(1 + (2 * i));
            float p_top = float(top) / float(1 + (2 * i));
            float p_bottom = float(bottom) / float(1 + (2 * i));

            if (p_top < 0.2 && p_bottom < 0.2)
                break;

            if (p_left < 0.2 && p_right < 0.2)
                break;
        }

        ellipse->size.height = radi;
        ellipse->size.width = radi;

        /*
        //collect points in threashold
        cv::Mat ch_mat=cv::Mat::zeros(pic->rows, pic->cols, CV_8UC1);
        cv::Point2i coor;
        std::vector<cv::Point2i> all_points;
        ch_mat.data[(pic->cols*(y0))+(x0)]=1;
        coor.x=x0;
        coor.y=y0;
        all_points.push_back(coor);
        int all_p_idx=0;
        while(all_p_idx<all_points.size()){
            cv::Point2i ak_p=all_points.at(all_p_idx);
            pic->data[(pic->cols*(ak_p.y))+(ak_p.x)]=255;
            for(int i=-1;i<2;i++)
                for(int j=-1;j<2;j++){
                    if(ak_p.y+j>0 && ak_p.y+j<pic->rows && ak_p.x+i>0 && ak_p.x+i<pic->cols)
                    if((int)ch_mat.data[(ch_mat.cols*(ak_p.y+j))+(ak_p.x+i)]==0 &&
                        (int)pic->data[(pic->cols*(ak_p.y+j))+(ak_p.x+i)]>th_down &&
                        (int)pic->data[(pic->cols*(ak_p.y+j))+(ak_p.x+i)]<th_up){
                            coor.x=ak_p.x+i;
                            coor.y=ak_p.y+j;
                            ch_mat.data[(pic->cols*(ak_p.y+j))+(ak_p.x+i)]=1;
                            all_points.push_back(coor);
                    }
                }
            all_p_idx++;
            //std::cout<<all_points.size()<<":"<<all_p_idx<<std::endl;
        }
        */
    }

    /**
    * @brief function to check if ellipse is good
    * @param ellipse ellipse to check
    * @param pic image to check
    * @param good_ellipse_threshold threshold for good ellipse
    * @param max_ellipse_radi max ellipse radius
    */
    static bool is_good_ellipse(cv::RotatedRect *ellipse, cv::Mat *pic, int good_ellipse_threshold, int max_ellipse_radi) {
        if (ellipse->center.x == 0 && ellipse->center.y == 0)
            return false;

        if (ellipse->size.width == 0 || ellipse->size.height == 0)
            grow_region(ellipse, pic, max_ellipse_radi);

        float x0 = ellipse->center.x;
        float y0 = ellipse->center.y;

        int st_x = x0 - (ellipse->size.width / 4.0);
        int st_y = y0 - (ellipse->size.height / 4.0);
        int en_x = x0 + (ellipse->size.width / 4.0);
        int en_y = y0 + (ellipse->size.height / 4.0);

        float val = 0.0;
        float val_cnt = 0;
        float ext_val = 0.0;

        for (int i = st_x; i < en_x; i++)
            for (int j = st_y; j < en_y; j++) {

                if (i > 0 && i < pic->cols && j > 0 && j < pic->rows) {
                    val += pic->data[(pic->cols * j) + i];
                    val_cnt++;
                }
            }

        if (val_cnt > 0)
            val = val / val_cnt;
        else
            return false;

        val_cnt = 0;

        st_x = x0 - (ellipse->size.width * 0.75);
        st_y = y0 - (ellipse->size.height * 0.75);
        en_x = x0 + (ellipse->size.width * 0.75);
        en_y = y0 + (ellipse->size.height * 0.75);

        int in_st_x = x0 - (ellipse->size.width / 2);
        int in_st_y = y0 - (ellipse->size.height / 2);
        int in_en_x = x0 + (ellipse->size.width / 2);
        int in_en_y = y0 + (ellipse->size.height / 2);

        for (int i = st_x; i < en_x; i++)
            for (int j = st_y; j < en_y; j++) {
                if (!(i >= in_st_x && i <= in_en_x && j >= in_st_y && j <= in_en_y))
                    if (i > 0 && i < pic->cols && j > 0 && j < pic->rows) {
                        ext_val += pic->data[(pic->cols * j) + i];
                        val_cnt++;
                        //pic->at<char>(j,i)=255;
                    }
            }

        if (val_cnt > 0)
            ext_val = ext_val / val_cnt;
        else
            return false;

        val = ext_val - val;

        if (val > good_ellipse_threshold)
            return true;
        else
            return false;
    }

    /**
    * @brief function to calculte rays in order to find ellipse
    * @param th_edges image to check
    * @param end_x max x
    * @param end_y max y
    * @param pos position to check
    * @param ret array to store results
    */
    static void rays(cv::Mat *th_edges, int end_x, int end_y, cv::Point *pos, int *ret) {

        for (int i = 0; i < 8; i++)
            ret[i] = -1;
        for (int i = 0; i < end_x; i++)
            for (int j = 0; j < end_y; j++) {

                if (pos->x - i > 0 && pos->x + i < th_edges->cols && pos->y - j > 0 && pos->y + j < th_edges->rows) {

                    if ((int) th_edges->data[(th_edges->cols * (pos->y)) + (pos->x + i)] != 0 && ret[0] == -1) {
                        ret[0] = th_edges->data[(th_edges->cols * (pos->y)) + (pos->x + i)] - 1;
                        //std::cout<<"val:"<<ret[0]<<" x:"<<pos->x+i<<" y:"<<pos->y<<std::endl;
                    }
                    if ((int) th_edges->data[(th_edges->cols * (pos->y)) + (pos->x - i)] != 0 && ret[1] == -1) {
                        ret[1] = th_edges->data[(th_edges->cols * (pos->y)) + (pos->x - i)] - 1;
                        //std::cout<<"val:"<<ret[0]<<" x:"<<pos->x-i<<" y:"<<pos->y<<std::endl;
                    }
                    if ((int) th_edges->data[(th_edges->cols * (pos->y + j)) + (pos->x)] != 0 && ret[2] == -1) {
                        ret[2] = th_edges->data[(th_edges->cols * (pos->y + j)) + (pos->x)] - 1;
                        //std::cout<<"val:"<<ret[0]<<" x:"<<pos->x<<" y:"<<pos->y+j<<std::endl;
                    }
                    if ((int) th_edges->data[(th_edges->cols * (pos->y - j)) + (pos->x)] != 0 && ret[3] == -1) {
                        ret[3] = th_edges->data[(th_edges->cols * (pos->y - j)) + (pos->x)] - 1;
                        //std::cout<<"val:"<<ret[0]<<" x:"<<pos->x<<" y:"<<pos->y-j<<std::endl;
                    }

                    if ((int) th_edges->data[(th_edges->cols * (pos->y + j)) + (pos->x + i)] != 0 && ret[4] == -1 &&
                        i == j) {
                        ret[4] = th_edges->data[(th_edges->cols * (pos->y + j)) + (pos->x + i)] - 1;
                        //std::cout<<"val:"<<ret[0]<<" x:"<<pos->x+i<<" y:"<<pos->y+j<<std::endl;
                    }
                    if ((int) th_edges->data[(th_edges->cols * (pos->y - j)) + (pos->x - i)] != 0 && ret[5] == -1 &&
                        i == j) {
                        ret[5] = th_edges->data[(th_edges->cols * (pos->y - j)) + (pos->x - i)] - 1;
                        //std::cout<<"val:"<<ret[0]<<" x:"<<pos->x-i<<" y:"<<pos->y-j<<std::endl;
                    }
                    if ((int) th_edges->data[(th_edges->cols * (pos->y - j)) + (pos->x + i)] != 0 && ret[6] == -1 &&
                        i == j) {
                        ret[6] = th_edges->data[(th_edges->cols * (pos->y - j)) + (pos->x + i)] - 1;
                        //std::cout<<"val:"<<ret[0]<<" x:"<<pos->x+i<<" y:"<<pos->y-j<<std::endl;
                    }
                    if ((int) th_edges->data[(th_edges->cols * (pos->y + j)) + (pos->x - i)] != 0 && ret[7] == -1 &&
                        i == j) {
                        ret[7] = th_edges->data[(th_edges->cols * (pos->y + j)) + (pos->x - i)] - 1;
                        //std::cout<<"val:"<<ret[0]<<" x:"<<pos->x-i<<" y:"<<pos->y+j<<std::endl;
                    }
                }
            }
    }

    /**
    * @brief function to find ellipse in GPU
    * @param cols number of columns
    * @param rows number of rows
    * @param th threshold value
    * @param edge_to_th distance from edge to threshold
    * @param mean_dist mean distance between points
    * @param area area to check
    * @param pos position to check
    */
    static void
    zero_around_region_th_bordergpu(int cols, int rows, int th, int edge_to_th, double mean_dist, double area,
                                    cv::RotatedRect *pos) {
        cudaMemset(d_th_edges, 0, cols * rows * sizeof(unsigned char));
        std::vector <cv::Point> selected_points;
        cv::RotatedRect ellipse;

        int start_x = pos->center.x - (area * cols);
        int end_x = pos->center.x + (area * cols);
        int start_y = pos->center.y - (area * rows);
        int end_y = pos->center.y + (area * rows);

        if (start_x < 0)
            start_x = edge_to_th;
        if (start_y < 0)
            start_y = edge_to_th;
        if (end_x > cols)
            end_x = cols - (edge_to_th + 1);
        if (end_y > rows)
            end_y = rows - (edge_to_th + 1);

        cv::Point st_pos;
        st_pos.x = pos->center.x;
        st_pos.y = pos->center.y;

        th = th + th + 1;

        // Calculation of edges from threshold image
        calculateBorderArroundTh<<<(cols * rows) / 128 + 1, 128>>>(d_pic, d_edges, d_th_edges, start_x, end_x,
                                                                start_y, end_y, cols, rows, th,
                                                                edge_to_th);
        cudaMemcpy(d_edges, d_th_edges, cols * rows * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
        // Create image with just the lines that are curved
        get_curvesgpu(cols, rows, start_x, end_x, start_y, end_y, mean_dist, 0);

        // Calculate rays and get the values
        cudaMemset(d_ret, 0x1F, 8 * sizeof(int));
        calculateRays<<<8, 128>>>(d_edges, (end_x - start_x) / 2, (end_y - start_y) / 2, st_pos.x, st_pos.y, d_ret,
                                cols, rows);
        cudaMemset(_atomicIndexRaysPoints, 0, sizeof(int));
        cudaMemset(d_xx, 0, sizeof(int) * cols * rows);
        cudaMemset(d_yy, 0, sizeof(int) * cols * rows);
        getPointsOfRaysCurves<<<(cols * rows) / 128 + 1, 128>>>(d_edges, d_ret, d_xx, d_yy, cols, rows);

        int size;
        cudaMemcpy(&size, _atomicIndexRaysPoints, sizeof(int), cudaMemcpyDeviceToHost);

    //    cout << "size:" << size << endl;

        if (size > 5)
            *pos = GPUfitEllipse(size);

        /*
        std::cout<<pos->x<<";"<<pos->y<<std::endl;
        cv::ellipse(*pic, cv::RotatedRect(*pos, cv::Size2f(5,5),0), CV_RGB(255,255,255));
        imshow("opt",*pic);
        */
    }

    /**
    * @brief function to find ellipse in CPU
    * @param pic original image to check
    * @param edges edge image to check
    * @param th_edges threshold image to check
    * @param th threshold value
    * @param edge_to_th distance from edge to threshold
    * @param mean_dist mean distance between points
    * @param area area to check
    * @param pos position to check
    */
    static void
    zero_around_region_th_border(cv::Mat *pic, cv::Mat *edges, cv::Mat *th_edges, int th, int edge_to_th, double mean_dist,
                                double area, cv::RotatedRect *pos) {

        int ret[8];
        std::vector <cv::Point> selected_points;
        cv::RotatedRect ellipse;

        int start_x = pos->center.x - (area * pic->cols);
        int end_x = pos->center.x + (area * pic->cols);
        int start_y = pos->center.y - (area * pic->rows);
        int end_y = pos->center.y + (area * pic->rows);

        if (start_x < 0)
            start_x = edge_to_th;
        if (start_y < 0)
            start_y = edge_to_th;
        if (end_x > pic->cols)
            end_x = pic->cols - (edge_to_th + 1);
        if (end_y > pic->rows)
            end_y = pic->rows - (edge_to_th + 1);

        th = th + th + 1;

        //std::cout<<"sx:"<<start_x<<" sy:"<<start_y<<" ex:"<<end_x<<" ey:"<<end_y<<" dist:"<<edge_to_th<<std::endl;
        for (int i = start_x; i < end_x; i++)
            for (int j = start_y; j < end_y; j++) {

                if (pic->data[(pic->cols * j) + (i)] < th) {

                    for (int k1 = -edge_to_th; k1 < edge_to_th; k1++)
                        for (int k2 = -edge_to_th; k2 < edge_to_th; k2++) {

                            if (i + k1 >= 0 && i + k1 < pic->cols && j + k2 > 0 && j + k2 < edges->rows)
                                if ((int) edges->data[(edges->cols * (j + k2)) + (i + k1)])
                                    th_edges->data[(edges->cols * (j + k2)) + (i + k1)] = 255;
                        }
                }
            }
        //remove_points_with_low_angle(th_edges, start_x, end_x, start_y, end_y);
        //show pic
    //    cv::imshow("th_edges", *th_edges);

        std::vector <std::vector<cv::Point>> all_curves = get_curves(pic, th_edges, start_x, end_x, start_y, end_y,
                                                                    mean_dist, 0);

    //    std::cout << "all curves:" << all_curves.size() << std::endl;

        if (all_curves.size() > 0) {
            //zero th_edges
            /*
        for(int i=start_x-edge_to_th; i<end_x+edge_to_th; i++)
            for(int j=start_y-edge_to_th; j<end_y+edge_to_th; j++){
                th_edges->data[(th_edges->cols*(j))+(i)]=0;
            }
            */

            for (int i = 0; i < th_edges->cols; i++)
                for (int j = 0; j < th_edges->rows; j++) {
                    th_edges->data[(th_edges->cols * (j)) + (i)] = 0;
                }

            //draw remaining edges
            for (int i = 0; i < all_curves.size(); i++) {
                //std::cout<<"written:"<<i+1<<std::endl;
                for (int j = 0; j < all_curves[i].size(); j++) {

                    if (all_curves[i][j].x >= 0 && all_curves[i][j].x < th_edges->cols && all_curves[i][j].y >= 0 &&
                        all_curves[i][j].y < th_edges->rows)
                        th_edges->data[(th_edges->cols * (all_curves[i][j].y)) + (all_curves[i][j].x)] =
                                i + 1; //+1 becouse of first is 0
                }
            }

            cv::Point st_pos;
            st_pos.x = pos->center.x;
            st_pos.y = pos->center.y;
            //send rays add edges to vector
            rays(th_edges, (end_x - start_x) / 2, (end_y - start_y) / 2, &st_pos, ret);

    //        for (int i = 0; i < 8; i++) std::cout << "ret:" << ret[i] << std::endl;
    //        cv::imshow("akt", *th_edges);

            //gather points
            for (int i = 0; i < 8; i++)
                if (ret[i] > -1 && ret[i] < all_curves.size()) {
                    //std::cout<<"size:"<<all_curves.size()<<std::endl;
                    //std::cout<<"idx:"<<ret[i]<<std::endl;
                    for (int j = 0; j < all_curves[ret[i]].size(); j++) {
                        selected_points.push_back(all_curves[ret[i]][j]);
                    }
                }
            //ellipse fit if size>5

    //        cout << "size: " << selected_points.size() << endl;

            if (selected_points.size() > 5) {

                *pos = cv::fitEllipse(cv::Mat(selected_points));
                /*
                cv::ellipse(*pic, cv::RotatedRect(ellipse.operator CvBox2D()),CV_RGB(255,255,255));
                cv::imshow("akt", *pic);
                */
            }
        }

        /*
        std::cout<<pos->x<<";"<<pos->y<<std::endl;
        cv::ellipse(*pic, cv::RotatedRect(*pos, cv::Size2f(5,5),0), CV_RGB(255,255,255));
        imshow("opt",*pic);
        */
    }


    /**
    * @brief optimize_pos in GPU
    * @param cols number of columns
    * @param rows number of rows
    * @param area area to search
    * @param pos position to optimize
    */
    static void optimize_posgpu(int cols, int rows, double area, cv::Point *pos) {
        int pos_x_cpu, pos_y_cpu, pos_count_cpu, min;
        int max_val = 100000;
        int start_x = pos->x - (area * cols);
        int end_x = pos->x + (area * cols);
        int start_y = pos->y - (area * rows);
        int end_y = pos->y + (area * rows);

        int reg_size = sqrt(sqrt(pow(double(area * cols * 2), 2) + pow(double(area * rows * 2), 2)));

        if (start_x < reg_size)
            start_x = reg_size;
        if (start_y < reg_size)
            start_y = reg_size;
        if (end_x > cols)
            end_x = cols - (reg_size + 1);
        if (end_y > rows)
            end_y = rows - (reg_size + 1);

        cudaMemset(_pos_x, 0, sizeof(int));
        cudaMemset(_pos_y, 0, sizeof(int));
        cudaMemset(_pos_count, 0, sizeof(int));
        cudaMemcpy(_min_val, &max_val, sizeof(int), cudaMemcpyHostToDevice);

        optimizePos<<<(cols * rows) / 128 + 1, 128>>>(d_pic, d_min, start_x, end_x, start_y, end_y, reg_size,
                                                    cols, rows);

        searchPos<<<(cols * rows) / 128 + 1, 128>>>(d_min, start_x, end_x, start_y, end_y, cols,
                                                    rows);

        cudaMemcpy(&pos_x_cpu, _pos_x, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pos_y_cpu, _pos_y, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&pos_count_cpu, _pos_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&min, _min_val, sizeof(int), cudaMemcpyDeviceToHost);

    //    cout << "min: " << min << endl;

        if (pos_count_cpu > 0) {
            pos->x = pos_x_cpu / pos_count_cpu;
            pos->y = pos_y_cpu / pos_count_cpu;
        }
    }

    /**
    * @brief optimize_pos in CPU
    * @param pic original picture
    * @param area area to search
    * @param pos position to optimize
    */
    static void optimize_pos(cv::Mat *pic, double area, cv::Point *pos) {

        int start_x = pos->x - (area * pic->cols);
        int end_x = pos->x + (area * pic->cols);
        int start_y = pos->y - (area * pic->rows);
        int end_y = pos->y + (area * pic->rows);

        int val;
        int min_akt;
        int min_val = 1000000;

        int pos_x = 0;
        int pos_y = 0;
        int pos_count = 0;

        int reg_size = sqrt(sqrt(pow(double(area * pic->cols * 2), 2) + pow(double(area * pic->rows * 2), 2)));

        if (start_x < reg_size)
            start_x = reg_size;
        if (start_y < reg_size)
            start_y = reg_size;
        if (end_x > pic->cols)
            end_x = pic->cols - (reg_size + 1);
        if (end_y > pic->rows)
            end_y = pic->rows - (reg_size + 1);

        for (int i = start_x; i < end_x; i++)
            for (int j = start_y; j < end_y; j++) {

                min_akt = 0;

                for (int k1 = -reg_size; k1 < reg_size; k1++)
                    for (int k2 = -reg_size; k2 < reg_size; k2++) {

                        if (i + k1 > 0 && i + k1 < pic->cols && j + k2 > 0 && j + k2 < pic->rows) {
                            val = (pic->data[(pic->cols * j) + (i)] - pic->data[(pic->cols * (j + k2)) + (i + k1)]);
                            if (val > 0)
                                min_akt += val;
                        }
                    }

                if (min_akt == min_val) {
                    pos_x += i;
                    pos_y += j;
                    pos_count++;
                }

                if (min_akt < min_val) {
                    min_val = min_akt;
                    pos_x = i;
                    pos_y = j;
                    pos_count = 1;
                }
            }

    //    cout << "min: " << min_val << endl;

        if (pos_count > 0) {
            pos->x = pos_x / pos_count;
            pos->y = pos_y / pos_count;
        }
    }

    /**
    * @brief runexcuse in CPU
    * @param pic original picture
    * @param pic_th thresholded picture
    * @param th_edges edges of thresholded picture
    * @param good_ellipse_threshold threshold for good ellipse
    * @param max_ellipse_radi max ellipse radius
    * @return ellipse
    */
    static cv::RotatedRect
    runexcuse(cv::Mat *pic, cv::Mat *pic_th, cv::Mat *th_edges, int good_ellipse_threshold, int max_ellipse_radi) {
        cv::normalize(*pic, *pic, 0, 255, cv::NORM_MINMAX, CV_8U);

        double border = 0.05;
        int peek_detector_factor = 10;
        int bright_region_th = 200;
        double mean_dist = 3;
        int inner_color_range = 5;
        double th_histo = 0.5;
        int max_region_hole = 5;
        int min_region_size = 7;
        double area_opt = 0.1;
        double area_edges = 0.2;
        int edge_to_th = 5;

        cv::RotatedRect ellipse;
        cv::Point pos(0, 0);

        int start_x = floor(double(pic->cols) * border);
        int start_y = floor(double(pic->rows) * border);

        int end_x = pic->cols - start_x;
        int end_y = pic->rows - start_y;

        double stddev = 0;
        bool edges_only_tried = false;
        bool peek_found = false;
        int threshold_up = 0;

        peek_found = peek(pic, &stddev, start_x, end_x, start_y, end_y, peek_detector_factor, bright_region_th);
    
        threshold_up = ceil(stddev / 2);
        threshold_up--;

        cv::Mat picpic = cv::Mat::zeros(end_y - start_y, end_x - start_x, CV_8U);

        for (int i = 0; i < picpic.cols; i++)
            for (int j = 0; j < picpic.rows; j++) {
                picpic.data[(picpic.cols * j) + i] = pic->data[(pic->cols * (start_y + j)) + (start_x + i)];
            }
        cv::Mat detected_edges2 = canny_impl(&picpic);

        cv::Mat detected_edges = cv::Mat::zeros(pic->rows, pic->cols, CV_8U);
        for (int i = 0; i < detected_edges2.cols; i++)
            for (int j = 0; j < detected_edges2.rows; j++) {
                detected_edges.data[(detected_edges.cols * (start_y + j)) + (start_x + i)] = detected_edges2.data[
                        (detected_edges2.cols * j) + i];
            }
    
        remove_points_with_low_angle(&detected_edges, start_x, end_x, start_y, end_y);
    
        if (peek_found) {
            edges_only_tried = true;
            ellipse = find_best_edge(pic, &detected_edges, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);

            if (ellipse.center.x <= 0 || ellipse.center.x >= pic->cols || ellipse.center.y <= 0 ||
                ellipse.center.y >= pic->rows) {
                ellipse.center.x = 0;
                ellipse.center.y = 0;
                ellipse.angle = 0.0;
                ellipse.size.height = 0.0;
                ellipse.size.width = 0.0;
                peek_found = false;
            }
        }
    
        if (!peek_found) {
            pos = th_angular_histo(pic, pic_th, start_x, end_x, start_y, end_y, threshold_up, th_histo, max_region_hole,
                                min_region_size);
            ellipse.center.x = pos.x;
            ellipse.center.y = pos.y;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;
        }

        if (pos.x == 0 && pos.y == 0 && !edges_only_tried) {
            ellipse = find_best_edge(pic, &detected_edges, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);
            peek_found = true;
        }
    
        if (pos.x > 0 && pos.y > 0 && pos.x < pic->cols && pos.y < pic->rows && !peek_found) {
            optimize_pos(pic, area_opt, &pos);
            ellipse.center.x = pos.x;
            ellipse.center.y = pos.y;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;

            zero_around_region_th_border(pic, &detected_edges, th_edges, threshold_up, edge_to_th, mean_dist, area_edges,
                                        &ellipse);
        }
        return ellipse;
    }

    /**
    * @brief initializeStructures Initialize all the structures needed for the CUDA functions
    * @param cols number of columns
    * @param rows number of rows
    * @param start_x start column position
    * @param end_x end column position
    * @param start_y start column position
    * @param end_y end column position
    */
    static void initializeStructures(int cols, int rows, int start_x, int end_x, int start_y, int end_y) {
        // Global variables
        cudaMalloc((void **) &d_pic, sizeof(unsigned char) * cols * rows);
        cudaMalloc((void **) &d_grayHist, sizeof(unsigned int) * 256);
        cudaMallocHost((void **) &h_grayHist, sizeof(unsigned int) * 256);
        cudaMalloc((void **) &d_meanFeld, sizeof(float) * (end_x - start_x));
        cudaMalloc((void **) &d_stdFeld, sizeof(float) * (end_x - start_x));

        cudaMalloc((void **) &d_smallPic, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        cudaMalloc((void **) &d_auxSmall, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        cudaMalloc((void **) &d_aux2Small, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        cudaMalloc((void **) &d_resX, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        cudaMalloc((void **) &d_resY, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        cudaMalloc((void **) &d_strong, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        cudaMalloc((void **) &d_weak, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        cudaMalloc((void **) &d_exitSmall, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        cudaMalloc((void **) &d_edges, sizeof(unsigned char) * cols * rows);
        cudaMalloc((void **) &d_edgesAux, sizeof(unsigned char) * cols * rows);

        cudaMalloc((void **) &d_excentricity, sizeof(bool) * MAX_CURVES);
        cudaMalloc((void **) &d_outputImg, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_translation, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_sum_x, sizeof(unsigned int) * MAX_CURVES);
        cudaMalloc((void **) &d_sum_y, sizeof(unsigned int) * MAX_CURVES);
        cudaMalloc((void **) &d_total, sizeof(unsigned int) * MAX_CURVES);
        cudaMalloc((void **) &d_innerGray, sizeof(unsigned int) * MAX_CURVES);

        cudaMalloc((void **) &d_xx, sizeof(int) * cols * rows);
        cudaMalloc((void **) &d_yy, sizeof(int) * cols * rows);

        cudaMallocHost((void **) &h_x, 5 * 5 * sizeof(float));
        cudaMalloc((void **) &d_info, sizeof(int));
        cudaMalloc((void **) &d_a, 5 * cols * rows * sizeof(float));
        cudaMalloc((void **) &d_b, 5 * 5 *sizeof(float));
        cudaMalloc((void **) &d_x, 5 * 5 *sizeof(float));
        cudaMalloc((void **) &d_array, sizeof(float *));
        cudaMalloc((void **) &d_a2inv, 5 * 5 * sizeof(float));
        cudaMemcpy(d_array, &d_a2inv, sizeof(float *), cudaMemcpyHostToDevice);

        cudaMallocHost((void **) &h_hist_l, DEF_SIZE * sizeof(int));
        cudaMallocHost((void **) &h_hist_lb, DEF_SIZE * sizeof(int));
        cudaMallocHost((void **) &h_hist_b, DEF_SIZE * sizeof(int));
        cudaMallocHost((void **) &h_hist_br, DEF_SIZE * sizeof(int));

        cudaMalloc((void **) &d_hist_l, DEF_SIZE * sizeof(int));
        cudaMalloc((void **) &d_hist_lb, DEF_SIZE * sizeof(int));
        cudaMalloc((void **) &d_hist_b, DEF_SIZE * sizeof(int));
        cudaMalloc((void **) &d_hist_br, DEF_SIZE * sizeof(int));

        cudaMalloc((void **) &d_min, cols * rows * sizeof(unsigned int));

        cudaMalloc((void **) &d_th_edges, cols * rows * sizeof(unsigned char));
        cudaMalloc((void **) &d_ret, 8 * sizeof(int));

        // Constant memory
        int lowanglePrecalc[2][8] = {-1, 0, 1, -1, 1, -1, 0, 1,
                                    -1, -1, -1, 0, 0, 1, 1, 1,};
        int nmsPrecalc[8][8] = {0, 1, 1, 1, 0, -1, -1, -1,
                                1, 0, 1, 1, -1, 0, -1, -1,
                                1, 0, 1, -1, -1, 0, -1, 1,
                                0, -1, 1, -1, 0, 1, -1, 1,
                                0, 1, 1, 1, 0, -1, -1, -1,
                                1, 0, 1, 1, -1, 0, -1, -1,
                                1, 0, 1, -1, -1, 0, -1, 1,
                                0, -1, 1, -1, 0, 1, -1, 1,};
        int raysPrecalc[2][8] = {0, 0, 1, -1, 1, -1, -1, 1,
                                1, -1, 0, 0, 1, -1, 1, -1,};
        float gau[16] = {0.000000220358050f, 0.000007297256405f, 0.000146569312970f, 0.001785579770079f,
                        0.013193749090229f, 0.059130281094460f, 0.160732768610747f, 0.265003534507060f, 0.265003534507060f,
                        0.160732768610747f, 0.059130281094460f, 0.013193749090229f, 0.001785579770079f, 0.000146569312970f,
                        0.000007297256405f, 0.000000220358050f};
        float deriv_gau[16] = {-0.000026704586264f, -0.000276122963398f, -0.003355163265098f, -0.024616683775044f,
                            -0.108194751875585f,
                            -0.278368310241814f, -0.388430056419619f, -0.196732206873178f, 0.196732206873178f,
                            0.388430056419619f,
                            0.278368310241814f, 0.108194751875585f, 0.024616683775044f, 0.003355163265098f,
                            0.000276122963398f, 0.000026704586264f};

        cudaMemcpyToSymbol(_lowanglePrecalc, lowanglePrecalc, 2 * 8 * sizeof(int));
        cudaMemcpyToSymbol(_nmsPrecalc, nmsPrecalc, sizeof(int) * 8 * 8);
        cudaMemcpyToSymbol(_raysPrecalc, raysPrecalc, sizeof(int) * 2 * 8);
        cudaMemcpyToSymbol(_gauC, gau, sizeof(float) * 16);
        cudaMemcpyToSymbol(_deriv_gauC, deriv_gau, sizeof(float) * 16);

        // Cublas handle
        cublasCreate(&d_handle);
        cublasCreate(&d_handlePeek);

        // Streams
        cudaStreamCreate(&d_stream_1);
        cudaStreamCreate(&d_stream_2);
        cublasSetStream_v2(d_handlePeek, d_stream_2);

        // Address from symbols
        cudaGetSymbolAddress((void **) &_d_sum_y, sum_y);
        cudaGetSymbolAddress((void **) &_d_sum_x, sum_x);
        cudaGetSymbolAddress((void **) &_translationIndex, translationIndex);
        cudaGetSymbolAddress((void **) &_atomicInnerGrayIndex, atomicInnerGrayIndex);
        cudaGetSymbolAddress((void **) &_atomicIndexRaysPoints, atomicIndexRaysPoints);
        cudaGetSymbolAddress((void **) &_min_val, min_val);
        cudaGetSymbolAddress((void **) &_pos_x, pos_x);
        cudaGetSymbolAddress((void **) &_pos_y, pos_y);
        cudaGetSymbolAddress((void **) &_pos_count, pos_count);
    }


    /**
    * @brief Free all the memory allocated in the GPU
    */
    static void freeStructures() {
        cudaFree(d_pic);
        cudaFree(d_grayHist);
        cudaFreeHost(h_grayHist);
        cudaFree(d_meanFeld);
        cudaFree(d_stdFeld);
        cudaFree(d_auxSmall);
        cudaFree(d_aux2Small);
        cudaFree(d_resX);
        cudaFree(d_resY);
        cudaFree(d_strong);
        cudaFree(d_weak);
        cudaFree(d_exitSmall);
        cudaFree(d_edges);
        cudaFree(d_edgesAux);
        cudaFree(d_excentricity);
        cudaFree(d_outputImg);
        cudaFree(d_translation);
        cudaFree(d_sum_x);
        cudaFree(d_sum_y);
        cudaFree(d_total);
        cudaFree(d_innerGray);
        cudaFree(d_xx);
        cudaFree(d_yy);
        cudaFree(d_info);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_a2inv);
        cudaFree(d_array);
        cudaFree(d_hist_b);
        cudaFree(d_hist_br);
        cudaFree(d_hist_l);
        cudaFree(d_hist_lb);
        cudaFree(d_min);
        cudaFree(d_th_edges);
        cudaFree(d_ret);

        cudaFreeHost(h_x);
        cudaFreeHost(h_hist_b);
        cudaFreeHost(h_hist_br);
        cudaFreeHost(h_hist_l);
        cudaFreeHost(h_hist_lb);

        cublasDestroy(d_handle);
        cublasDestroy(d_handlePeek);

        cudaStreamDestroy(d_stream_1);
        cudaStreamDestroy(d_stream_2);

        cudaDeviceReset();
    }

    /**
    * @brief runexcuse in GPU
    * @param pic input image
    * @param pic_th thresholded image
    * @param th_edges thresholded edges
    * @param good_ellipse_threshold
    * @param max_ellipse_radi
    * @return ellipse
    */
    static cv::RotatedRect
    runexcusegpu(cv::Mat *pic, cv::Mat *pic_th, cv::Mat *th_edges, int good_ellipse_threshold, int max_ellipse_radi,
                int iteration) {
        cv::normalize(*pic, *pic, 0, 255, cv::NORM_MINMAX, CV_8U);

        double border = 0.1;
        int peek_detector_factor = 10;
        int bright_region_th = 200;
        double mean_dist = 3;
        int inner_color_range = 5;
        double th_histo = 0.5;
        int max_region_hole = 5;
        int min_region_size = 7;
        double area_opt = 0.1;
        double area_edges = 0.2;
        int edge_to_th = 5;

        cv::RotatedRect ellipse;
        cv::Point pos(0, 0);

        int start_x = floor(double(pic->cols) * border);
        int start_y = floor(double(pic->rows) * border);

        int end_x = pic->cols - start_x;
        int end_y = pic->rows - start_y;

        double stddev = 0;
        bool edges_only_tried = false;
        bool peek_found = false;
        int threshold_up = 0;

        int rows = pic->rows;
        int cols = pic->cols;

        if (iteration == 0) {
            initializeStructures(cols, rows, start_x, end_x, start_y, end_y);
        }
        cudaMemcpy(d_pic, pic->data, sizeof(unsigned char) * cols * rows, cudaMemcpyHostToDevice);
        peek_found = peekgpu(cols, rows, &stddev, start_x, end_x, start_y, end_y, peek_detector_factor, bright_region_th);
        threshold_up = ceil(stddev / 2);
        threshold_up--;

        cudaMemset(d_smallPic, 0, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        removeBorders<<<cols * rows / 256 + 1, 256>>>(d_pic, d_smallPic, cols, rows,
                                                    start_x, end_x, start_y, end_y);

        canny_impl_gpu((end_x - start_x), (end_y - start_y));

        cudaMemset(d_edges, 0, sizeof(unsigned char) * cols * rows);
        addBorders<<<(rows * cols) / 256 + 1, 256>>>(d_exitSmall, d_edges, cols, rows, start_x, end_x,
                                                    start_y, end_y);
        
        int start_x_n = start_x + 5;
        int end_x_n = end_x - 5;
        int start_y_n = start_y + 5;
        int end_y_n = end_y - 5;

        if (start_x_n < 5)
            start_x_n = 5;
        if (end_x_n > cols - 5)
            end_x_n = cols - 5;
        if (start_y_n < 5)
            start_y_n = 5;
        if (end_y_n > rows - 5)
            end_y_n = rows - 5;

        remove_points_with_low_anglegpu(cols, rows, start_x_n, end_x_n, start_y_n, end_y_n);
        
        if (peek_found) {
            edges_only_tried = true;
            ellipse = find_best_edgegpu(cols, rows, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);
            if (ellipse.center.x <= 0 || ellipse.center.x >= pic->cols || ellipse.center.y <= 0 ||
                ellipse.center.y >= pic->rows) {
                ellipse.center.x = 0;
                ellipse.center.y = 0;
                ellipse.angle = 0.0;
                ellipse.size.height = 0.0;
                ellipse.size.width = 0.0;
                peek_found = false;
            }
        }
        
        if (!peek_found) {
            pos = th_angular_histogpu(cols, rows, start_x, end_x, start_y, end_y, threshold_up, th_histo, max_region_hole,
                                    min_region_size);
            ellipse.center.x = pos.x;
            ellipse.center.y = pos.y;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;
        }

        if (pos.x == 0 && pos.y == 0 && !edges_only_tried) {
            ellipse = find_best_edgegpu(cols, rows, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);
            peek_found = true;
        }
        
        // Reset edges to original edges
        cudaMemcpy(d_edges, d_edgesAux, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
        if (pos.x > 0 && pos.y > 0 && pos.x < cols && pos.y < rows && !peek_found) {
            optimize_posgpu(cols, rows, area_opt, &pos);
            ellipse.center.x = pos.x;
            ellipse.center.y = pos.y;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;

            zero_around_region_th_bordergpu(cols, rows, threshold_up, edge_to_th, mean_dist, area_edges,
                                            &ellipse);
        }

        return ellipse;
    }

    /**
    * @brief Pupilrun
    * @param frame
    * @return pupil ellipse
    */
    Pupil run(const Mat &frame) {

        Mat downscaled = frame;
        float scalingRatio = 1.0;
        if (frame.rows > IMG_SIZE || frame.cols > IMG_SIZE) {
            // return ellipse;
            // Downscaling
            float rw = IMG_SIZE / (float) frame.cols;
            float rh = IMG_SIZE / (float) frame.rows;
            scalingRatio = min<float>(min<float>(rw, rh), 1.0);
            cv::resize(frame, downscaled, Size(), scalingRatio, scalingRatio, INTER_LINEAR);
        }

        Mat target;
        normalize(downscaled, target, 0, 255, NORM_MINMAX, CV_8U);

        Mat pic_th = Mat::zeros(target.rows, target.cols, CV_8U);
        Mat th_edges = Mat::zeros(target.rows, target.cols, CV_8U);

        cv::RotatedRect ellipse = runexcuse(&target, &pic_th, &th_edges, good_ellipse_threshold, max_ellipse_radi);
        cv::RotatedRect scaledEllipse(cv::Point2f(ellipse.center.x / scalingRatio, ellipse.center.y / scalingRatio),
                                    cv::Size2f(ellipse.size.width / scalingRatio, ellipse.size.height / scalingRatio),
                                    ellipse.angle);

        return Pupil(scaledEllipse);
    }

    /**
    * @brief Pupilrungpu
    * @param frame
    * @param iteration
    * @return pupil ellipse
    */
    Pupil rungpu(const Mat &frame, int iteration) {

        Mat downscaled = frame;
        float scalingRatio = 1.0;
        if (frame.rows > IMG_SIZE || frame.cols > IMG_SIZE) {
            // return ellipse;
            // Downscaling
            float rw = IMG_SIZE / (float) frame.cols;
            float rh = IMG_SIZE / (float) frame.rows;
            scalingRatio = min<float>(min<float>(rw, rh), 1.0);
            cv::resize(frame, downscaled, Size(), scalingRatio, scalingRatio, INTER_LINEAR);
        }

        Mat target;
        normalize(downscaled, target, 0, 255, NORM_MINMAX, CV_8U);

        Mat pic_th = Mat::zeros(target.rows, target.cols, CV_8U);
        Mat th_edges = Mat::zeros(target.rows, target.cols, CV_8U);

        cv::RotatedRect ellipse = runexcusegpu(&target, &pic_th, &th_edges, good_ellipse_threshold, max_ellipse_radi,
                                            iteration);
        cv::RotatedRect scaledEllipse(cv::Point2f(ellipse.center.x / scalingRatio, ellipse.center.y / scalingRatio),
                                    cv::Size2f(ellipse.size.width / scalingRatio, ellipse.size.height / scalingRatio),
                                    ellipse.angle);

        return Pupil(scaledEllipse);
    }
}

extern "C" Pupil EXCUSEGREEDYII_run(const Mat &frame, int iteration, int gpu) {
    if (gpu) {
        return ExcuseGreedyII::rungpu(frame, iteration);
    } else {
        return ExcuseGreedyII::run(frame);
    }
}