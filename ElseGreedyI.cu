/*
  Version 1.0, 17.12.2015, Copyright University of Tübingen.
  The code and the algorithm are for non-comercial use only.

  The code is parallelized using CUDA and CUBLAS by Arturo Vicente Jaén. 14/04/2023. Copyrigth University of Murcia.
  This file contains the CUDA kernels and functions for the pupil detection working together for the parallelized final version.
*/

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <opencv2/opencv.hpp>
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

#include "include/Pupil.h"

using namespace cv;
using namespace std;


namespace ElseGreedyI {
    float minArea = 0;
    float maxArea = 0;

    float minAreaRatio = 0.005;
    float maxAreaRatio = 0.2;

    #define IMG_SIZE 1100
    #define MAX_LINE 37
    #define MAX_LINE_CPU 10000
    #define KERNEL_SIZE 16
    #define HALF_KERNEL_SIZE 8
    #define MAX_CURVES 50

//--------------------------------------------- VARIABLE DEFINITIONS -------------------------------------------------

    // Constant memory
    __device__ __constant__ int _nmsPrecalc[8][8];
    __device__ __constant__ int _lowanglePrecalc[2][8];
    __device__ __constant__ float _gauC[16];
    __device__ __constant__ float _deriv_gauC[16];


    // Global memory
    unsigned int *d_grayHist, *d_innerGray, *d_innerGrayCount, *d_outputImg, *d_translation, *d_sum_x, *d_sum_y, *d_total, *d_sum_xx, *d_sum_yy, *d_totall;
    float *d_smallPic, *d_auxSmall, *d_aux2Small, *d_resX, *d_resY, *d_strong, *d_weak, *d_exitSmall, *h_x, *d_img, *d_result, *d_resultNeg;
    unsigned char *d_pic, *d_edges, *d_edgesAux;
    bool *d_excentricity;
    int *d_yy, *d_info;
    float *d_bb, *d_AA, *d_AAinv, *d_xx;
    int *d_A_index;
    float **aa_bb, **aa_AA, **aa_AAinv, **aa_xx;

    // Cublas handle
    cublasHandle_t d_handle;
    // Cuda symbols directions
    float *_d_sum_y, *_d_sum_x, *_translationIndex, *_bestEdge;

    // 2d Filters
    float *_conv;
    float *_convNeg;

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
    * @brief kernel for applying a deriv gau (Sobel) 1D filter by rows (in the middle of the kernel as anchor (-1,-1)) USING BORDER REPLICATION
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
    * @brief kernel for applying a deriv gau (Sobel) 1D filter by columns (in the middle of the kernel as anchor (-1,-1)) USING BORDER REPLICATION
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
    * @brief kernel for removing points with low angle in the second step (Uses just diagonal parallelism) USING SHARED MEMORY
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
    remove_points_with_low_angle_gpu_2_diagonalShmem(unsigned char *src, unsigned char *dst, int start_x, int end_x,
                                                    int start_y,
                                                    int end_y, int width, int height) {
        int index = threadIdx.x;
        int numDiag = width + height - 1;
        int diagonalSize = min(width, height);

        extern __shared__ int shmem[];
        // Copy first, second and third diagonals to shared memory
        if (index == 0) {
            shmem[0] = (int) src[0];
            shmem[diagonalSize] = (int) src[1];
            shmem[diagonalSize + 1] = (int) src[width];
            shmem[2 * diagonalSize] = (int) src[2];
            shmem[2 * diagonalSize + 1] = (int) src[width + 1];
            shmem[2 * diagonalSize + 2] = (int) src[2 * width];
        }
        // Initialize loop, for each diagonal, load the next diagonal to shared memory and compute current
        for (int k = 2; k < numDiag - 1; k++) {
            int i = index + start_x;
            int j = k - index + start_y;
            int idx = j * width + i;

            // Bring next diagonal to shared memory
            if ((i < end_y) && (j + 1 < end_x) && (j + 1 >= start_x) && (index < diagonalSize - 1)) {
                shmem[((k + 1) % 3) * diagonalSize + index] = (int) src[idx + 1];
            }

            __syncthreads();

            if ((i < end_y) && (j < end_x) && (j >= start_x) && (index < diagonalSize - 1) && (index > 0)) {
                if (shmem[(k % 3) * diagonalSize + index]) {
                    shmem[(k % 3) * diagonalSize + index] *= !((shmem[((k + 2) % 3) * diagonalSize + index] &&
                                                                shmem[((k + 1) % 3) * diagonalSize + index + 1]) ||
                                                            (shmem[((k + 2) % 3) * diagonalSize + index - 1] &&
                                                                shmem[((k + 2) % 3) * diagonalSize + index]) ||
                                                            (shmem[((k + 2) % 3) * diagonalSize + index - 1] &&
                                                                shmem[((k + 1) % 3) * diagonalSize + index]) ||
                                                            (shmem[((k + 1) % 3) * diagonalSize + index] &&
                                                                shmem[((k + 1) % 3) * diagonalSize + index + 1]));
                }
                src[idx] = (unsigned char) shmem[(k % 3) * diagonalSize + index];
            }
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
                if (expresion1) {
                    dst[(j + 1) * width + (i)] = 255;
                }

                bool expresion2 = ((box[14] && !box[7] && !box[10]) && ((box[8] || box[6]) && (box[16] || box[15])));
                dst[(j + 1) * width + (i + 1)] *= !expresion2;
                dst[(j + 1) * width + (i - 1)] *= !expresion2;
                dst[(j + 2) * width + (i + 1)] *= !expresion2;
                dst[(j + 2) * width + (i - 1)] *= !expresion2;
                if (expresion2) {
                    dst[(j + 1) * width + (i)] = 255;
                    dst[(j + 2) * width + (i)] = 255;
                }

                bool expresion3 = ((box[9] && !box[5]) && (box[8] || box[2]));
                dst[(j + 1) * width + (i + 1)] *= !expresion3;
                dst[(j - 1) * width + (i + 1)] *= !expresion3;
                if (expresion3) {
                    dst[(j)*width + (i + 1)] = 255;
                }

                bool expresion4 = ((box[11] && !box[5] && !box[9]) && ((box[8] || box[2]) && (box[13] || box[12])));
                dst[(j + 1) * width + (i + 1)] *= !expresion4;
                dst[(j - 1) * width + (i + 1)] *= !expresion4;
                dst[(j + 1) * width + (i + 2)] *= !expresion4;
                dst[(j - 1) * width + (i + 2)] *= !expresion4;
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

                int box2[18];
                box2[1] = (int) src[(j) * width + (i - 1)];

                box2[2] = (int) src[(j - 1) * width + (i - 2)];
                box2[3] = (int) src[(j - 2) * width + (i - 3)];

                box2[4] = (int) src[(j - 1) * width + (i + 1)];
                box2[5] = (int) src[(j - 2) * width + (i + 2)];

                box2[6] = (int) src[(j + 1) * width + (i - 2)];
                box2[7] = (int) src[(j + 2) * width + (i - 3)];

                box2[8] = (int) src[(j + 1) * width + (i + 1)];
                box2[9] = (int) src[(j + 2) * width + (i + 2)];

                box2[10] = (int) src[(j + 1) * width + (i)];

                box2[15] = (int) src[(j - 1) * width + (i - 1)];
                box2[16] = (int) src[(j - 2) * width + (i - 2)];

                box2[11] = (int) src[(j + 2) * width + (i + 1)];
                box2[12] = (int) src[(j + 3) * width + (i + 2)];

                box2[13] = (int) src[(j + 2) * width + (i - 1)];
                box2[14] = (int) src[(j + 3) * width + (i - 2)];

                if ((box2[1] && box2[2] && box2[3] && box2[4] && box2[5]) ||
                    (box2[1] && box2[6] && box2[7] && box2[8] && box2[9]) ||
                    (box2[10] && box2[11] && box2[12] && box2[4] && box2[5]) ||
                    (box2[10] && box2[13] && box2[14] && box2[15] && box2[16])) {
                    src[j * width + (i)] = 0;
                }
            }
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
                        int numCols, int numRows) {

        // Calculate index
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols && g_labels[id] > 0) {
            // translate label
            atomicAdd(&g_sum_x[g_labels[id]], id % numCols);
            atomicAdd(&g_sum_y[g_labels[id]], id / numCols);
            atomicAdd(&g_total[g_labels[id]], 1);
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
    calculateExcentricity(unsigned int *g_labels, unsigned int *g_sum_x, unsigned int *g_sum_y,
                        unsigned int *g_total, bool *excentricity, double mean_dist, int numCols, int numRows) {
        // Calculate index
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols && g_labels[id] > 0) {
            // translate label
            unsigned int label = g_labels[id];
            // calculate valid centroid
            float mean_x = floorf(fdividef(float(g_sum_x[label]), float(g_total[label])) + 0.5);
            float mean_y = floorf(fdividef(float(g_sum_y[label]), float(g_total[label])) + 0.5);
            int x = id % numCols;
            int y = id / numCols;
            if (excentricity[label] && (fabsf(mean_x - x) <= mean_dist && fabsf(mean_y - y) <= mean_dist))
                excentricity[label] = false;
        }
    }

    __device__ unsigned int translationIndex = 1;

    /**
    * @brief kernel for calculating the translation from sparse form of the labels to dense form
    * @param g_labels labels array
    * @param g_translation translation array
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__
    void calculateTranslation(unsigned int *g_labels, unsigned int *g_translation, bool *excentricity,
                            unsigned int *g_total, unsigned int *g_sum_x, unsigned int *g_sum_y,
                            unsigned int *g_totall, unsigned int *g_sum_xx, unsigned int *g_sum_yy,
                            double mean_dist, int numCols, int numRows) {
        // Calculate index
        const unsigned int id = ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols +
                                ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols && excentricity[g_labels[id]] && g_total[g_labels[id]] > 10) {
            // translate label
            if (atomicCAS(&g_translation[g_labels[id]], 0, 1) == 0) {
                int futureLabel = atomicAdd(&translationIndex, 1);
                if (futureLabel < MAX_CURVES) {
                    g_translation[g_labels[id]] = futureLabel;
                    g_totall[futureLabel] = g_total[g_labels[id]];
                    g_sum_xx[futureLabel] = g_sum_x[g_labels[id]];
                    g_sum_yy[futureLabel] = g_sum_y[g_labels[id]];
                }
            }
        }
    }

    /**
    * @brief kernel for calculating the A matrices of each label
    * @param g_labels array of labels
    * @param g_sum_x sum of x coordinates of each label
    * @param g_sum_y sum of y coordinates of each label
    * @param g_total sum of pixels of each label
    * @param g_translation translation array from sparse to dense form of labels
    * @param d_As array of A matrices
    * @param d_As_index array of number of pixels of each label
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    calculateAMatrices(unsigned int *g_labels, unsigned int *g_sum_x, unsigned int *g_sum_y, unsigned int *g_total,
                    unsigned int *g_translation, float *d_As, int *d_As_index, int numCols, int numRows) {
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);

        if (id < numRows * numCols && g_translation[g_labels[id]] > 0) {
            unsigned int label = g_translation[g_labels[id]];
            float mean_x = fdividef(g_sum_x[label], g_total[label]);
            float mean_y = fdividef(g_sum_y[label], g_total[label]);
            float x = (id % numCols) - mean_x;
            float y = (id / numCols) - mean_y;

            int index = atomicAdd(&d_As_index[label], 1);
            if (index >= 3 * MAX_LINE)
                return;
            d_As[label * 5 * 3 * MAX_LINE + index * 5] = x * x;
            d_As[label * 5 * 3 * MAX_LINE + index * 5 + 1] = x * y;
            d_As[label * 5 * 3 * MAX_LINE + index * 5 + 2] = y * y;
            d_As[label * 5 * 3 * MAX_LINE + index * 5 + 3] = x;
            d_As[label * 5 * 3 * MAX_LINE + index * 5 + 4] = y;
        }
    }

    /**
    * @brief kernel for calculating the B matrices of each label
    * @param d_Bs array of B matrices
    * @param d_As_index array of number of pixels of each label
    * @param d_As array of A matrices
    */
    __global__ void
    calculateBMatrices(float *d_Bs, int *d_As_index, float *d_As) {
        int column = blockIdx.x % 5;
        int label = blockIdx.x / 5;
        int index = threadIdx.x;
        if (label <= translationIndex)
            for (int i = index; i < d_As_index[label]; i += blockDim.x) {
                atomicAdd(&d_Bs[label * 5 + column], d_As[label * 5 * 3 * MAX_LINE + i * 5 + column]);
            }
    }

    /**
    * @brief kernel for calculating the C matrices of each label
    * @param pic original image
    * @param g_labels array of labels
    * @param g_sum_x sum of x coordinates of each label
    * @param g_sum_y sum of y coordinates of each label
    * @param g_total sum of pixels of each label
    * @param g_translation translation array from sparse to dense form of labels
    * @param g_inner_gray array of inner gray values of each label
    * @param g_inner_gray_count array of number of pixels of each label
    * @param g_xx array of x coordinates of each label
    * @param numCols number of columns
    * @param numRows number of rows
    */
    __global__ void
    calculateInnerGray2(unsigned char *pic, unsigned int *g_labels, unsigned int *g_sum_x, unsigned int *g_sum_y,
                        unsigned int *g_total, unsigned int *g_translation, unsigned int *g_inner_gray,
                        unsigned int *g_inner_gray_count, float *g_xx, int numCols, int numRows) {
        // Calculate index
        const unsigned int id =
                ((blockIdx.y * blockDim.y) + threadIdx.y) * numCols + ((blockIdx.x * blockDim.x) + threadIdx.x);
        if (id < numRows * numCols && g_labels[id] > 0 && g_translation[g_labels[id]] < translationIndex) {
            int label = g_translation[g_labels[id]];

            int x = id % numCols;
            int y = id / numCols;

            float centerEllipseX = g_xx[label * 5];
            float centerEllipseY = g_xx[label * 5 + 1];

            int vec_x = (int) roundf(x - centerEllipseX);
            int vec_y = (int) roundf(y - centerEllipseY);

            int inner_gray = 0;
            int inner_gray_count = 0;
    #pragma unroll
            for (float p = 0.95f; p > 0.80f; p -= 0.01f) {
                int p_x = (int) roundf(centerEllipseX + float((float(vec_x) * p) + 0.5));
                int p_y = (int) roundf(centerEllipseY + float((float(vec_y) * p) + 0.5));
                if (p_x < 0 || p_x >= numCols || p_y < 0 || p_y >= numRows)
                    continue;
                inner_gray += (unsigned int) pic[(numCols * (p_y)) + (p_x)];
                inner_gray_count++;
            }
            atomicAdd(&g_inner_gray[label], inner_gray);
            atomicAdd(&g_inner_gray_count[label], inner_gray_count);
        }
    }

    /**
    * @brief kernel for passing from exit of the equation system to the final ellipse parameters
    * @param g_sum_x sum of x coordinates of each label
    * @param g_sum_y sum of y coordinates of each label
    * @param g_total total number of pixels of each label
    * @param g_xx result of the equation system (also exit of the real ellipse parameters)
    */
    __global__
    void calculateEllipses(unsigned int *g_sum_x, unsigned int *g_sum_y, unsigned int *g_total, float *g_xx) {
        const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id > 0 && id < translationIndex) {
            float cos_phi, sin_phi, mean_x, mean_y, orientation_rad, a, c, d, e;

            float mean_xx = g_sum_x[id] / float(g_total[id]);
            float mean_yy = g_sum_y[id] / float(g_total[id]);

            float a1 = g_xx[id * 5];
            float b1 = g_xx[id * 5 + 1];
            float c1 = g_xx[id * 5 + 2];
            float d1 = g_xx[id * 5 + 3];
            float e1 = g_xx[id * 5 + 4];

            float orientationTolerance = 1e-3;

            if (fminf(fabsf(fdividef(b1, a1)), fabsf(fdividef(b1, c1))) > orientationTolerance) {
                orientation_rad = 0.5 * atanf(b1 / (c1 - a1));
                cos_phi = cosf(orientation_rad);
                sin_phi = sinf(orientation_rad);
                a = a1 * cos_phi * cos_phi - b1 * cos_phi * sin_phi + c1 * sin_phi * sin_phi;
                c = a1 * sin_phi * sin_phi + b1 * cos_phi * sin_phi + c1 * cos_phi * cos_phi;
                d = d1 * cos_phi - e1 * sin_phi;
                e = d1 * sin_phi + e1 * cos_phi;
                mean_x = cos_phi * mean_xx - sin_phi * mean_yy;
                mean_y = sin_phi * mean_xx + cos_phi * mean_yy;
            } else {
                orientation_rad = 0;
                cos_phi = cosf(orientation_rad);
                sin_phi = sinf(orientation_rad);
                a = a1;
                c = c1;
                d = d1;
                e = e1;
                mean_x = mean_xx;
                mean_y = mean_yy;
            }

            if (a * c <= 0) {
                g_xx[id * 5] = 0;
                g_xx[id * 5 + 1] = 0;
                g_xx[id * 5 + 2] = 0;
                g_xx[id * 5 + 3] = 0;
                g_xx[id * 5 + 4] = 0;
                return;
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

            if (F <= 0) {
                g_xx[id * 5] = 0;
                g_xx[id * 5 + 1] = 0;
                g_xx[id * 5 + 2] = 0;
                g_xx[id * 5 + 3] = 0;
                g_xx[id * 5 + 4] = 0;
                return;
            }

            float aa = sqrtf(F / a);
            float bb = sqrtf(F / c);

            // Apply rotation to center point
            float X0_in = cos_phi * X0 + sin_phi * Y0;
            float Y0_in = -sin_phi * X0 + cos_phi * Y0;

            g_xx[id * 5] = X0_in;
            g_xx[id * 5 + 1] = Y0_in;
            g_xx[id * 5 + 2] = 2 * min(aa, bb);
            g_xx[id * 5 + 3] = 2 * max(aa, bb);
            g_xx[id * 5 + 4] = -orientation_rad * 180 / float(M_PI) + 90 * (aa > bb);
        }
    }

    __device__ int bestLabel = 0;

    /**
    * @brief kernel for finding if the ellipse is good or not
    * @param g_xx array of ellipse parameters
    * @param pic original image
    * @param numCols number of columns of the image
    * @param numRows number of rows of the image
    * @return good or not good ellipse
    */
    __device__ bool isGoodEllipse(float *g_xx, unsigned char *pic, int numCols, int numRows) {
        if (g_xx[0] == 0 && g_xx[1] == 0)
            return false;

        float x0 = g_xx[0];
        float y0 = g_xx[1];
        float width = g_xx[2];
        float height = g_xx[3];

        int st_x = (int) ::ceilf(x0 - width / 4);
        int st_y = (int) ::ceilf(y0 - height / 4);

        int en_x = (int) ::floorf(x0 + width / 4);
        int en_y = (int) ::floorf(y0 + height / 4);

        float val = 0;
        int count = 0;
        float ext_val = 0;

        for (int y = max(0, st_y); y <= min(en_y, numRows - 1); y++) {
            for (int x = max(0, st_x); x <= min(en_x, numCols - 1); x++) {
                val += pic[y * numCols + x];
                count++;
            }
        }

        if (count == 0)
            return false;
        val /= count;

        count = 0;

        st_x = (int) (x0 - width * 0.75);
        st_y = (int) (y0 - height * 0.75);
        en_x = (int) (x0 + width * 0.75);
        en_y = (int) (y0 + height * 0.75);

        int in_st_x = (int) (x0 - width / 2);
        int in_st_y = (int) (y0 - height / 2);
        int in_en_x = (int) (x0 + width / 2);
        int in_en_y = (int) (y0 + height / 2);

    //    for (int y = max(0, st_y); y <= min(en_y, numRows - 1); y++) {
    //        for (int x = max(0, st_x); x <= min(en_x, numCols - 1); x++) {
    //            if (!(x >= in_st_x && x <= in_en_x && y >= in_st_y && y <= in_en_y)) {
    //                ext_val += pic[y * numCols + x];
    //                count++;
    //            }
    //        }
    //    }

    // remove if's, loop over 4 different zones
        for (int y = max(0, st_y); y < in_st_y; y++) {
            for (int x = max(0, st_x); x <= min(en_x, numCols - 1); x++) {
                ext_val += pic[y * numCols + x];
                count++;
            }
        }

        for (int y = in_st_y; y <= in_en_y; y++) {
            for (int x = max(0, st_x); x < in_st_x; x++) {
                ext_val += pic[y * numCols + x];
                count++;
            }
            for (int x = in_en_x; x <= min(en_x, numCols - 1); x++) {
                ext_val += pic[y * numCols + x];
                count++;
            }
        }

        for (int y = in_en_y; y <= min(en_y, numRows - 1); y++) {
            for (int x = max(0, st_x); x <= min(en_x, numCols - 1); x++) {
                ext_val += pic[y * numCols + x];
                count++;
            }
        }

        if (count == 0)
            return false;

        ext_val /= count;

        if (ext_val - val < 10)
            return false;

        return true;
    }

    /**
    * @brief kernel for finding if the ellipses are good or not
    * @param pic original image
    * @param g_xx array of ellipse parameters
    * @param numCols number of columns of the image
    * @param numRows number of rows of the image
    */
    __global__ void
    areGoodEllipses(unsigned char *pic, float *g_xx, int numCols, int numRows) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;

        if (id < translationIndex) {
            float x0 = g_xx[id * 5];
            float y0 = g_xx[id * 5 + 1];
            float width = g_xx[id * 5 + 2];
            float height = g_xx[id * 5 + 3];

            if (x0 < 0 || y0 < 0 || width < 0 || height < 0 || x0 >= numCols || y0 >= numRows ||
                height > 3 * width || width > 3 * height ||
                width * height < numCols * numRows * 0.005 || width * height > numCols * numRows * 0.2 ||
                !isGoodEllipse(g_xx + id * 5, pic, numCols, numRows)) {
                g_xx[id * 5] = 0;
                g_xx[id * 5 + 1] = 0;
                g_xx[id * 5 + 2] = 0;
                g_xx[id * 5 + 3] = 0;
                g_xx[id * 5 + 4] = 0;
                return;
            }
        }
    }

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
    __global__ void
    selectBestCurve(unsigned char *pic, unsigned int *g_total, unsigned int *g_inner_gray, unsigned int *g_inned_gray_count,
                    float *g_xx, int numCols, int numRows) {
        int bestSize = 0;
        int bestGray = 1000000;
        bestLabel = 0;

        for (int i = 1; i < translationIndex; i++) {
            float x0 = g_xx[i * 5];
            float y0 = g_xx[i * 5 + 1];
            float width = g_xx[i * 5 + 2];
            float height = g_xx[i * 5 + 3];

            if (x0 == 0 && y0 == 0 && width == 0 && height == 0)
                continue;

            int gray_val = (int) (g_inner_gray[i] / g_inned_gray_count[i]);
            if (g_inned_gray_count[i] == 0) gray_val = 1000;

            int inner = gray_val * (1 + abs(height - width));

            if (inner < bestGray) {
                bestGray = inner;
                bestSize = g_total[i];
                bestLabel = i;
            } else if (inner == bestGray && g_total[i] > bestSize) {
                bestGray = inner;
                bestSize = g_total[i];
                bestLabel = i;
            }
        }
    }

    //----------------------------------------------------- BLOB FINDER -----------------------------------------------------------------------------
    /**
    * @brief Kernel for reduce size of the image for calculating the blob finder by using mean of tile
    * @param src image to reduce
    * @param dst reduced image
    * @param fak factor to reduce
    * @param cols width of the image
    * @param rows height of the image
    */
    __global__
    void mumgpu(unsigned char *src, float *dst, int fak, int cols, int rows) {
        int fak_ges = fak + 1;
        int sz_x = cols / fak_ges;
        int sz_y = rows / fak_ges;
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < sz_x * sz_y) {
            int i = index / sz_x;
            int j = index % sz_x;
            int idx = (j + 1) * fak_ges;
            int idy = (i + 1) * fak_ges;
            int hist[256];
    #pragma unroll 256
            for (int ii = 0; ii < 256; ii++)
                hist[ii] = 0;

            int mean = 0;
            int cnt = 0;

            for (int ii = -fak; ii <= fak; ii++)
                for (int jj = -fak; jj <= fak; jj++) {
                    if (idy + ii > 0 && idy + ii < rows && idx + jj > 0 && idx + jj < cols) {
                        if ((unsigned int) src[(cols * (idy + ii)) + (idx + jj)] > 255)
                            src[(cols * (idy + ii)) + (idx + jj)] = 255;
                        hist[src[(cols * (idy + ii)) + (idx + jj)]]++;
                        cnt++;
                        mean += src[(cols * (idy + ii)) + (idx + jj)];
                    }
                }
            mean = mean / cnt;

            int mean_2 = 0;
            cnt = 0;
            for (int ii = 0; ii <= mean; ii++) {
                mean_2 += ii * hist[ii];
                cnt += hist[ii];
            }

            if (cnt == 0)
                mean_2 = mean;
            else
                mean_2 = mean_2 / cnt;

            dst[(sz_x * (i)) + (j)] = float(mean_2);
        }
    }

    /**
    * @brief Kernel for performing 2d convolution
    * @param src image to convolve
    * @param dst convolved image
    * @param width number of columns
    * @param height number of rows
    * @param conv_size size of the convolution kernel (size of a side as it is a square kernel)
    * @param conv convolution kernel
    */
    __global__
    void gau2DConv(float *src, float *dst, int width, int height, int conv_size, float *conv) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;
        int half_conv_size = conv_size / 2;
        if (i < height && j < width) {
            float sum = 0;
    #pragma unroll
            for (int yprim = 0; yprim < conv_size; yprim++) {
                for (int xprim = 0; xprim < conv_size; xprim++) {
                    int ii = i + yprim - half_conv_size;
                    int jj = j + xprim - half_conv_size;
                    if (i + yprim - half_conv_size < 0)
                        ii = 0;
                    if (i + yprim - half_conv_size >= height)
                        ii = height - 1;
                    if (j + xprim - half_conv_size < 0)
                        jj = 0;
                    if (j + xprim - half_conv_size >= width)
                        jj = width - 1;
                    sum += src[ii * width + jj] * conv[yprim * conv_size + xprim];
                }
            }
            dst[i * width + j] = sum;
        }
    }

    /**
    * @brief Kernel for multiplying two exits of the convolution (positive and negative)
    * @param dst result of the multiplication
    * @param result image of the positive convolution
    * @param resultNeg image of the negative convolution
    * @param width number of columns
    * @param height number of rows
    */
    __global__
    void multiply(float *dst, float *result, float *resultNeg, int width, int height) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int i = index / width;
        int j = index % width;
        if (i < height && j < width)
            dst[i * width + j] = result[i * width + j] * !(result[i * width + j] < 0) * (255.0f - resultNeg[i * width + j]);
    }

    //-------------------------------------------------- END OF CUDA KERNELS ------------------------------------------------------------------------

    /**
    * @brief function to check if the ellipse is good
    * @param ellipse ellipse to check
    * @param pic picture to check
    * @param erg pointer to the result
    * @return true if the ellipse is good
    */
    static bool is_good_ellipse_eval(RotatedRect *ellipse, Mat *pic, int *erg) {

        if (ellipse->center.x == 0 && ellipse->center.y == 0)
            return false;

        float x0 = ellipse->center.x;
        float y0 = ellipse->center.y;

        int st_x = (int) ceil(x0 - (ellipse->size.width / 4.0));
        int st_y = (int) ceil(y0 - (ellipse->size.height / 4.0));
        int en_x = (int) floor(x0 + (ellipse->size.width / 4.0));
        int en_y = (int) floor(y0 + (ellipse->size.height / 4.0));

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

        st_x = (int) (x0 - (ellipse->size.width * 0.75));
        st_y = (int) (y0 - (ellipse->size.height * 0.75));
        en_x = (int) (x0 + (ellipse->size.width * 0.75));
        en_y = (int) (y0 + (ellipse->size.height * 0.75));

        int in_st_x = (int) ceil(x0 - (ellipse->size.width / 2));
        int in_st_y = (int) ceil(y0 - (ellipse->size.height / 2));
        int in_en_x = (int) floor(x0 + (ellipse->size.width / 2));
        int in_en_y = (int) floor(y0 + (ellipse->size.height / 2));

        for (int i = st_x; i < en_x; i++)
            for (int j = st_y; j < en_y; j++) {
                if (!(i >= in_st_x && i <= in_en_x && j >= in_st_y && j <= in_en_y))
                    if (i > 0 && i < pic->cols && j > 0 && j < pic->rows) {
                        ext_val += pic->data[(pic->cols * j) + i];
                        val_cnt++;
                    }
            }

        if (val_cnt > 0)
            ext_val = ext_val / val_cnt;
        else
            return false;

        val = ext_val - val;

        *erg = (int) val;

        if (val > 10)
            return true;
        else
            return false;
    }

    /**
    * @brief function to calculate the inner gray value of the ellipse in CPU
    * @param pic image to check
    * @param curve curve of the ellipse
    * @param ellipse ellipse to check
    * @return inner gray value
    */
    static int calc_inner_gray(Mat *pic, std::vector <Point> curve, RotatedRect ellipse) {

        int gray_val = 0;
        int gray_cnt = 0;

        Mat checkmap = Mat::zeros(pic->size(), CV_8U);

        for (unsigned int i = 0; i < curve.size(); i++) {

            int vec_x = (int) round(curve[i].x - ellipse.center.x);
            int vec_y = (int) round(curve[i].y - ellipse.center.y);

            for (float p = 0.95f; p > 0.80f; p -= 0.01f) { //0.75;-0.05
                int p_x = (int) round(ellipse.center.x + float((float(vec_x) * p) + 0.5));
                int p_y = (int) round(ellipse.center.y + float((float(vec_y) * p) + 0.5));

                if (p_x > 0 && p_x < pic->cols && p_y > 0 && p_y < pic->rows) {

                    if (checkmap.data[(pic->cols * p_y) + p_x] == 0) {
                        checkmap.data[(pic->cols * p_y) + p_x] = 1;
                        gray_val += (unsigned int) pic->data[(pic->cols * p_y) + p_x];
                        gray_cnt++;
                    }
                }
            }
        }

        if (gray_cnt > 0)
            gray_val = gray_val / gray_cnt;
        else
            gray_val = 1000;

        return gray_val;
    }

    /**
    * @brief function to get the curves from the edge image and the most eliptic and dark inside curve in GPU
    * @param pic the original image
    * @param edge the edge image
    * @param start_x the start column
    * @param end_x the end column
    * @param start_y the start row
    * @param end_y the end row
    * @param mean_dist the mean distance between the centroid and the curve
    * @param inner_color_range the range of the inner color
    **/
    static RotatedRect
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

        cudaMemset(d_sum_x, 0, cols * rows * sizeof(unsigned int));
        cudaMemset(d_sum_y, 0, cols * rows * sizeof(unsigned int));
        cudaMemset(d_total, 0, cols * rows * sizeof(unsigned int));
        calculateCentroid<<<grid, block>>>(d_outputImg, d_sum_x, d_sum_y, d_total, cols, rows);

        cudaMemset(d_excentricity, true, cols * rows * sizeof(bool));

        calculateExcentricity<<<grid, block>>>(d_outputImg, d_sum_x, d_sum_y, d_total, d_excentricity,
                                            mean_dist, cols, rows);

        cudaMemset(d_translation, 0, cols * rows * sizeof(unsigned int));

        unsigned int uno = 1;
        cudaMemcpy(_translationIndex, &uno, sizeof(unsigned int), cudaMemcpyHostToDevice);

        cudaMemset(d_sum_xx, 0, MAX_CURVES * sizeof(unsigned int));
        cudaMemset(d_sum_yy, 0, MAX_CURVES * sizeof(unsigned int));
        cudaMemset(d_totall, 0, MAX_CURVES * sizeof(unsigned int));
        calculateTranslation<<<grid, block>>>(d_outputImg, d_translation, d_excentricity,
                                            d_total, d_sum_x, d_sum_y,
                                            d_totall, d_sum_xx, d_sum_yy,
                                            mean_dist, cols, rows);

        int translation = 0;
        cudaMemcpy(&translation, _translationIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (translation == 0) {
            cv::RotatedRect ellipse;
            ellipse.center.x = 0;
            ellipse.center.y = 0;
            ellipse.angle = 0.0;
            ellipse.size.height = 0.0;
            ellipse.size.width = 0.0;
            return ellipse;
        }

        cudaMemset(d_innerGray, 0, MAX_CURVES * sizeof(int));
        cudaMemset(d_innerGrayCount, 0, MAX_CURVES * sizeof(int));

        cudaMemset(d_bb, 0, MAX_CURVES * 5 * sizeof(float));
        cudaMemset(d_AA, 0, MAX_CURVES * 3 * MAX_LINE * 5 * sizeof(float));
        cudaMemset(d_AAinv, 0, MAX_CURVES * 5 * 5 * sizeof(float));
        cudaMemset(d_xx, 0, MAX_CURVES * 5 * sizeof(float));
        cudaMemset(d_A_index, 0, MAX_CURVES * sizeof(int));

        calculateAMatrices<<<grid, block>>>(d_outputImg, d_sum_xx, d_sum_yy, d_totall, d_translation, d_AA,
                                            d_A_index, cols, rows);

        calculateBMatrices<<<(translation) * 5, 256>>>(d_bb, d_A_index, d_AA);

        float alpha = 1.0f;
        float beta = 0.0f;

        // Calculate covariance matrix
        cublasSgemmBatched(d_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, 3 * MAX_LINE, &alpha, aa_AA, 5, aa_AA, 5, &beta, aa_AA,
                        5,
                        (translation));

        // Calculate (A^t * A) ^ -1 A^T
        cublasSgemmBatched(d_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, 5, &alpha, aa_AA, 5, aa_AA, 5, &beta, aa_AAinv, 5,
                        (translation));

    //     Calculate (A^t * A)^-1
        cublasSmatinvBatched(d_handle, 5, aa_AAinv, 5, aa_AAinv, 5, d_info, (translation));

    //     Calculate (A^t * A)^-1 * A^t
        cublasSgemmBatched(d_handle, CUBLAS_OP_N, CUBLAS_OP_T, 5, 5, 5, &alpha, aa_AAinv, 5, aa_AA, 5, &beta, aa_AA, 5,
                        (translation));
        // Calculate (A^t * A)^-1 * A^t * B
        cublasSgemmBatched(d_handle, CUBLAS_OP_N, CUBLAS_OP_N, 5, 1, 5, &alpha, aa_AA, 5, aa_bb, 5, &beta, aa_xx, 5,
                        (translation));

        calculateEllipses<<<translation / 64 + 1, 64>>>(d_sum_xx, d_sum_yy, d_totall, d_xx);

        calculateInnerGray2<<<grid, block>>>(d_pic, d_outputImg, d_sum_xx, d_sum_yy, d_totall, d_translation,
                                            d_innerGray, d_innerGrayCount, d_xx, cols, rows);

        areGoodEllipses<<<translation / 64 + 1, 64>>>(d_pic, d_xx, cols, rows);

        selectBestCurve<<<1, 1>>>(d_pic, d_totall, d_innerGray, d_innerGrayCount, d_xx, cols,
                                rows);

        int bestCurve;
        cudaMemcpy(&bestCurve, _bestEdge, sizeof(int), cudaMemcpyDeviceToHost);

        cv::RotatedRect ellipse;
        ellipse.center.x = 0;
        ellipse.center.y = 0;
        ellipse.angle = 0.0;
        ellipse.size.height = 0.0;
        ellipse.size.width = 0.0;

        if (bestCurve > 0) {
            cudaMemcpy(h_x, d_xx + bestCurve * 5, 5 * sizeof(float), cudaMemcpyDeviceToHost);
            ellipse.center.x = h_x[0];
            ellipse.center.y = h_x[1];
            ellipse.angle = h_x[4];
            ellipse.size.width = h_x[2];
            ellipse.size.height = h_x[3];
        }

    //    unsigned int *h_outputImg = (unsigned int *) malloc(cols * rows * sizeof(unsigned int));
    //    cudaMemcpy(h_outputImg, d_outputImg, cols * rows * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //    Mat outputImg = Mat(rows, cols, CV_32SC1, h_outputImg);
    //    Mat outputImg2 = Mat(rows, cols, CV_8UC1, cv::Scalar(0));
    //    for (int i = 0; i < rows; i++) {
    //        for (int j = 0; j < cols; j++) {
    //            if (h_outputImg[i * cols + j] > 0) {
    //                outputImg2.at<uchar>(i, j) = outputImg.at<unsigned int>(i, j) % 240 + 15;
    //            }
    //        }
    //    }
    //    //set colormap jet
    //    applyColorMap(outputImg2, outputImg2, COLORMAP_JET);
    //    imshow("outputImg", outputImg2);

    //    getPointsOfBestCurve<<<grid, block>>>(d_outputImg, d_translation, d_edges, cols, rows);

        return ellipse;
    }

    /**
    * @brief function to get the curves from the edge image and the most eliptic and dark inside curve
    * @param pic image to get the curves from
    * @param edge edge image of the original image
    * @param magni magni image (it is not used but included in the original version)
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param mean_dist mean distance between the centroid and the curve
    * @param inner_color_range range of the inner color
    * @return selected curves
    */
    static std::vector <std::vector<Point>>
    get_curves(Mat *pic, Mat *edge, Mat *magni, int start_x, int end_x, int start_y, int end_y, double mean_dist,
            int inner_color_range) {

        (void) magni;
        std::vector <std::vector<Point>> all_lines;

        std::vector <std::vector<Point>> all_curves;
        std::vector <Point> curve;

        std::vector <Point> all_means;

        if (start_x < 2)
            start_x = 2;
        if (start_y < 2)
            start_y = 2;
        if (end_x > pic->cols - 2)
            end_x = pic->cols - 2;
        if (end_y > pic->rows - 2)
            end_y = pic->rows - 2;

        int curve_idx = 0;
        Point mean_p;
        bool add_curve;
        int mean_inner_gray;
        int mean_inner_gray_last = 1000000;

        all_curves.clear();
        all_means.clear();
        all_lines.clear();

        bool check[IMG_SIZE][IMG_SIZE];

        for (int i = 0; i < IMG_SIZE; i++)
            for (int j = 0; j < IMG_SIZE; j++)
                check[i][j] = 0;

        //get all lines
        for (int i = start_x; i < end_x; i++)
            for (int j = start_y; j < end_y; j++) {

                if (edge->data[(edge->cols * (j)) + (i)] > 0 && !check[i][j]) {
                    check[i][j] = 1;

                    curve.clear();
                    curve_idx = 0;

                    curve.push_back(Point(i, j));
                    mean_p.x = i;
                    mean_p.y = j;
                    curve_idx++;

                    int akt_idx = 0;

                    while (akt_idx < curve_idx) {

                        Point akt_pos = curve[akt_idx];
                        for (int k1 = -1; k1 < 2; k1++)
                            for (int k2 = -1; k2 < 2; k2++) {

                                if (akt_pos.x + k1 >= start_x && akt_pos.x + k1 < end_x && akt_pos.y + k2 >= start_y &&
                                    akt_pos.y + k2 < end_y)
                                    if (!check[akt_pos.x + k1][akt_pos.y + k2])
                                        if (edge->data[(edge->cols * (akt_pos.y + k2)) + (akt_pos.x + k1)] > 0) {
                                            check[akt_pos.x + k1][akt_pos.y + k2] = 1;

                                            mean_p.x += akt_pos.x + k1;
                                            mean_p.y += akt_pos.y + k2;
                                            curve.push_back(Point(akt_pos.x + k1, akt_pos.y + k2));
                                            curve_idx++;
                                        }
                            }
                        akt_idx++;
                    }

                    if (curve_idx > 10 && curve.size() > 10) {

                        mean_p.x = (int) floor((double(mean_p.x) / double(curve_idx)) + 0.5);
                        mean_p.y = (int) floor((double(mean_p.y) / double(curve_idx)) + 0.5);

                        all_means.push_back(mean_p);
                        all_lines.push_back(curve);
                    }
                }
            }

        RotatedRect selected_ellipse;

        for (unsigned int iii = 0; iii < all_lines.size(); iii++) {

            curve = all_lines.at(iii);
            mean_p = all_means.at(iii);

            int results = 0;
            add_curve = true;

            RotatedRect ellipse;

            for (unsigned int i = 0; i < curve.size(); i++)
                if (abs(mean_p.x - curve[i].x) <= mean_dist && abs(mean_p.y - curve[i].y) <= mean_dist)
                    add_curve = false;

            //is ellipse fit possible
            if (add_curve) {

                ellipse = fitEllipse(Mat(curve));

                if (ellipse.center.x < 0 || ellipse.center.y < 0 ||
                    ellipse.center.x > pic->cols || ellipse.center.y > pic->rows) {

                    add_curve = false;
                }

                if (ellipse.size.height > 3 * ellipse.size.width ||
                    ellipse.size.width > 3 * ellipse.size.height) {

                    add_curve = false;
                }

                if (add_curve) { // pupil area
                    if (ellipse.size.width * ellipse.size.height < minArea ||
                        ellipse.size.width * ellipse.size.height > maxArea)
                        add_curve = false;
                }

                if (add_curve) {
                    if (!is_good_ellipse_eval(&ellipse, pic, &results))
                        add_curve = false;
                }
            }

            if (add_curve) {

                if (inner_color_range >= 0) {
                    mean_inner_gray = 0;
                    mean_inner_gray = calc_inner_gray(pic, curve, ellipse);
                    mean_inner_gray = (int) (mean_inner_gray * (1 + abs(ellipse.size.height - ellipse.size.width)));

                    if (mean_inner_gray_last > mean_inner_gray) {
                        mean_inner_gray_last = mean_inner_gray;

                        all_curves.clear();
                        all_curves.push_back(curve);
                    } else if (mean_inner_gray_last == mean_inner_gray) {

                        if (curve.size() > all_curves[0].size()) {
                            mean_inner_gray_last = mean_inner_gray;
                            all_curves.clear();
                            all_curves.push_back(curve);
                            selected_ellipse = ellipse;
                        }
                    }
                }
            }
        }

        return all_curves;
    }

    /**
    * @brief function for finding the best edge in GPU
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
    find_best_edgegpu(int cols, int rows, int start_x, int end_x, int start_y, int end_y, double mean_dist,
                    int inner_color_range) {

        cv::RotatedRect ellipse;
        ellipse.center.x = 0;
        ellipse.center.y = 0;
        ellipse.angle = 0.0;
        ellipse.size.height = 0.0;
        ellipse.size.width = 0.0;

        ellipse = get_curvesgpu(cols, rows, start_x, end_x, start_y, end_y, mean_dist, inner_color_range);

    //    cudaMemset(_atomicInnerGrayIndex, 0, sizeof(int));
    //    cudaMemset(d_xx, 0, sizeof(float) * cols * rows);
    //    cudaMemset(d_yy, 0, sizeof(float) * cols * rows);
    //    getPointsOfInnerGrayCurves<<<(cols * rows) / 128 + 1, 128>>>(d_edges, d_xx, d_yy, cols, rows);

    //    unsigned char *h_edges = (unsigned char *) malloc(sizeof(unsigned char) * cols * rows);
    //    cudaMemcpy(h_edges, d_edges, sizeof(unsigned char) * cols * rows, cudaMemcpyDeviceToHost);
    //    Mat edges(rows, cols, CV_8UC1, h_edges);
    //    normalize(edges, edges, 0, 255, NORM_MINMAX, CV_8UC1);
    //    imshow("edges", edges);

    //    int curveSize;
    //    cudaMemcpy(&curveSize, _atomicInnerGrayIndex, sizeof(int), cudaMemcpyDeviceToHost);
    //
    //    if (curveSize > 4) {
    //        ellipse = GPUfitEllipse(curveSize);
    //
    //        //draw ellipse
    //        //cv::ellipse(*pic, ellipse, cv::Scalar(255, 0, 0), 2, 8);
    //        //show image
    //        //cv::imshow("ellipse", *pic);
    //
    //        if (ellipse.center.x < 0 || ellipse.center.y < 0 || ellipse.center.x > cols ||
    //            ellipse.center.y > rows) {
    //            ellipse.center.x = 0;
    //            ellipse.center.y = 0;
    //            ellipse.angle = 0.0;
    //            ellipse.size.height = 0.0;
    //            ellipse.size.width = 0.0;
    //        }
    //    } else {
    //        ellipse.center.x = 0;
    //        ellipse.center.y = 0;
    //        ellipse.angle = 0.0;
    //        ellipse.size.height = 0.0;
    //        ellipse.size.width = 0.0;
    //    }

        return ellipse;
    }

    /**
    * @brief function for finding the best edge in CPU
    * @param pic original picture
    * @param edge edge picture
    * @param magni magni image (it is not used but included in the original version)
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param mean_dist mean distance between points
    * @param inner_color_range range of inner color
    * @return best edge
    */
    static RotatedRect
    find_best_edge(Mat *pic, Mat *edge, Mat *magni, int start_x, int end_x, int start_y, int end_y, double mean_dist,
                int inner_color_range) {

        RotatedRect ellipse;
        ellipse.center.x = 0;
        ellipse.center.y = 0;
        ellipse.angle = 0.0;
        ellipse.size.height = 0.0;
        ellipse.size.width = 0.0;

        std::vector <std::vector<Point>> all_curves = get_curves(pic, edge, magni, start_x, end_x, start_y, end_y,
                                                                mean_dist, inner_color_range);

        if (all_curves.size() == 1) {
            ellipse = fitEllipse(Mat(all_curves[0]));

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
    * @param edge edge picture
    * @param start_x start column
    * @param end_x end column
    * @param start_y start row
    * @param end_y end row
    * @param mean_dist mean distance between points
    * @param inner_color_range range of inner color
    */
    static void filter_edgesgpu(int cols, int rows, int start_xx, int end_xx, int start_yy, int end_yy) {

        remove_points_with_low_angle_gpu_2_diagonal<<<1, 512>>>(d_edges, d_edges, start_xx, end_xx,
                                                                start_yy, end_yy, cols, rows);
        cudaMemcpy(d_edgesAux, d_edges, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);


        remove_points_with_low_angle_gpu_3<<<(rows * cols) / 256 + 1, 256>>>(d_edges, d_edgesAux, start_xx, end_xx,
                                                                            start_yy, end_yy,
                                                                            cols, rows);
        cudaMemcpy(d_edges, d_edgesAux, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
        remove_points_with_low_angle_gpu_4<<<(rows * cols) / 256 + 1, 256>>>(d_edges, d_edgesAux, start_xx, end_xx,
                                                                            start_yy, end_yy,
                                                                            cols, rows);
        cudaMemcpy(d_edges, d_edgesAux, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToDevice);

    //    unsigned char *h_edges = (unsigned char *) malloc(rows * cols * sizeof(unsigned char));
    //    cudaMemcpy(h_edges, d_edges, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //    Mat edges(rows, cols, CV_8UC1, h_edges);
    //    imshow("edges", edges);
    }

    /**
    * @brief function for filtering the edges using morphological operations
    * @param edge edge picture
    * @param start_xx start column
    * @param end_xx end column
    * @param start_yy start row
    * @param end_yy end row
    */
    static void filter_edges(Mat *edge, int start_xx, int end_xx, int start_yy, int end_yy) {

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
                }
            }

        //too many neigbours
        for (int j = start_y; j < end_y; j++)
            for (int i = start_x; i < end_x; i++) {
                int neig = 0;

                for (int k1 = -1; k1 < 2; k1++)
                    for (int k2 = -1; k2 < 2; k2++) {

                        if (edge->data[(edge->cols * (j + k1)) + (i + k2)] > 0)
                            neig++;
                    }

                if (neig > 3)
                    edge->data[(edge->cols * (j)) + (i)] = 0;
            }

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

                    if (box[0] && box[25] && box[2] && box[27])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[0] && box[25] && box[6] && box[28])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[8] && box[26] && box[2] && box[27])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box[8] && box[26] && box[6] && box[28])
                        edge->data[(edge->cols * (j)) + (i)] = 0;

                    int box2[18];
                    box2[1] = (int) edge->data[(edge->cols * (j)) + (i - 1)];

                    box2[2] = (int) edge->data[(edge->cols * (j - 1)) + (i - 2)];
                    box2[3] = (int) edge->data[(edge->cols * (j - 2)) + (i - 3)];

                    box2[4] = (int) edge->data[(edge->cols * (j - 1)) + (i + 1)];
                    box2[5] = (int) edge->data[(edge->cols * (j - 2)) + (i + 2)];

                    box2[6] = (int) edge->data[(edge->cols * (j + 1)) + (i - 2)];
                    box2[7] = (int) edge->data[(edge->cols * (j + 2)) + (i - 3)];

                    box2[8] = (int) edge->data[(edge->cols * (j + 1)) + (i + 1)];
                    box2[9] = (int) edge->data[(edge->cols * (j + 2)) + (i + 2)];

                    box2[10] = (int) edge->data[(edge->cols * (j + 1)) + (i)];

                    box2[15] = (int) edge->data[(edge->cols * (j - 1)) + (i - 1)];
                    box2[16] = (int) edge->data[(edge->cols * (j - 2)) + (i - 2)];

                    box2[11] = (int) edge->data[(edge->cols * (j + 2)) + (i + 1)];
                    box2[12] = (int) edge->data[(edge->cols * (j + 3)) + (i + 2)];

                    box2[13] = (int) edge->data[(edge->cols * (j + 2)) + (i - 1)];
                    box2[14] = (int) edge->data[(edge->cols * (j + 3)) + (i - 2)];

                    if (box2[1] && box2[2] && box2[3] && box2[4] && box2[5])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box2[1] && box2[6] && box2[7] && box2[8] && box2[9])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box2[10] && box2[11] && box2[12] && box2[4] && box2[5])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                    if (box2[10] && box2[13] && box2[14] && box2[15] && box2[16])
                        edge->data[(edge->cols * (j)) + (i)] = 0;
                }
            }
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
    * @param pic input image
    * @return binary edge  image
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
        hysteresis_pro<<<dim3(rows/16 + 1, cols/16 + 1), dim3(16, 16)>>>(d_strong, d_weak, d_exitSmall, cols, rows);

        return;
    }

    /**
    * @brief function for performing CANNY EDGE in CPU
    * @param pic input image
    * @return binary edge  image
    */
    static Mat canny_impl(Mat *pic, Mat *magni) {
        int k_sz = 16;

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

        Point anchor = Point(-1, -1);
        float delta = 0;
        int ddepth = -1;

        pic->convertTo(*pic, CV_32FC1);

        Mat gau_x = Mat(1, k_sz, CV_32FC1, &gau);
        Mat deriv_gau_x = Mat(1, k_sz, CV_32FC1, &deriv_gau);

        Mat res_x;
        Mat res_y;

        transpose(*pic, *pic);
        filter2D(*pic, res_x, ddepth, gau_x, anchor, delta, BORDER_REPLICATE);
        transpose(*pic, *pic);
        transpose(res_x, res_x);

        filter2D(res_x, res_x, ddepth, deriv_gau_x, anchor, delta, BORDER_REPLICATE);
        filter2D(*pic, res_y, ddepth, gau_x, anchor, delta, BORDER_REPLICATE);

        transpose(res_y, res_y);
        filter2D(res_y, res_y, ddepth, deriv_gau_x, anchor, delta, BORDER_REPLICATE);
        transpose(res_y, res_y);

        *magni = Mat::zeros(pic->rows, pic->cols, CV_32FC1);

        float *p_res, *p_x, *p_y;
        for (int i = 0; i < magni->rows; i++) {
            p_res = magni->ptr<float>(i);
            p_x = res_x.ptr<float>(i);
            p_y = res_y.ptr<float>(i);

            for (int j = 0; j < magni->cols; j++) {
                //res.at<float>(j, i)= sqrt( (res_x.at<float>(j, i)*res_x.at<float>(j, i)) + (res_y.at<float>(j, i)*res_y.at<float>(j, i)) );
                //res.at<float>(j, i)=robust_pytagoras_after_MOLAR_MORRIS(res_x.at<float>(j, i), res_y.at<float>(j, i));
                //res.at<float>(j, i)=hypot(res_x.at<float>(j, i), res_y.at<float>(j, i));

                //p_res[j]=__ieee754_hypot(p_x[j], p_y[j]);

                p_res[j] = std::hypot(p_x[j], p_y[j]);
            }
        }

        //th selection
        int PercentOfPixelsNotEdges = (int) round(0.7 * magni->cols * magni->rows);

        float high_th = 0;

        int h_sz = 64;
        int hist[64];
        for (int i = 0; i < h_sz; i++)
            hist[i] = 0;

        normalize(*magni, *magni, 0, 1, NORM_MINMAX, CV_32FC1);

        Mat res_idx = Mat::zeros(pic->rows, pic->cols, CV_8U);
        normalize(*magni, res_idx, 0, 63, NORM_MINMAX, CV_32S);

        int *p_res_idx = 0;
        for (int i = 0; i < magni->rows; i++) {
            p_res_idx = res_idx.ptr<int>(i);
            for (int j = 0; j < magni->cols; j++) {
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

        //non maximum supression + interpolation
        Mat non_ms = Mat::zeros(pic->rows, pic->cols, CV_8U);
        Mat non_ms_hth = Mat::zeros(pic->rows, pic->cols, CV_8U);

        float ix, iy, grad1, grad2, d;

        char *p_non_ms, *p_non_ms_hth;
        float *p_res_t, *p_res_b;
        for (int i = 1; i < magni->rows - 1; i++) {
            p_non_ms = non_ms.ptr<char>(i);
            p_non_ms_hth = non_ms_hth.ptr<char>(i);

            p_res = magni->ptr<float>(i);
            p_res_t = magni->ptr<float>(i - 1);
            p_res_b = magni->ptr<float>(i + 1);

            p_x = res_x.ptr<float>(i);
            p_y = res_y.ptr<float>(i);

            for (int j = 1; j < magni->cols - 1; j++) {

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

        //Mat res_lin = Mat::zeros(pic->rows, pic->cols, CV_8U);
        //bwselect(&non_ms_hth, &non_ms, &res_lin);

        Mat res_lin = cbwselect(non_ms_hth, non_ms);

        //res_lin.convertTo(res_lin, CV_8U);

        return res_lin;
    }

    /**
    * @brief Function to calculate reduced image with mean value of pixels
    * @param pic original image
    * @param result reduced image
    * @param fak factor to reduce image
    */
    static void mum(Mat *pic, Mat *result, int fak) {

        int fak_ges = fak + 1;
        int sz_x = pic->cols / fak_ges;
        int sz_y = pic->rows / fak_ges;

        *result = Mat::zeros(sz_y, sz_x, CV_8U);

        int hist[256];
        int mean = 0;
        int cnt = 0;
        int mean_2 = 0;

        int idx = 0;
        int idy = 0;

        for (int i = 0; i < sz_y; i++) {
            idy += fak_ges;

            for (int j = 0; j < sz_x; j++) {
                idx += fak_ges;

                for (int k = 0; k < 256; k++)
                    hist[k] = 0;

                mean = 0;
                cnt = 0;

                for (int ii = -fak; ii <= fak; ii++)
                    for (int jj = -fak; jj <= fak; jj++) {

                        if (idy + ii > 0 && idy + ii < pic->rows && idx + jj > 0 && idx + jj < pic->cols) {
                            if ((unsigned int) pic->data[(pic->cols * (idy + ii)) + (idx + jj)] > 255)
                                pic->data[(pic->cols * (idy + ii)) + (idx + jj)] = 255;

                            hist[pic->data[(pic->cols * (idy + ii)) + (idx + jj)]]++;
                            cnt++;
                            mean += pic->data[(pic->cols * (idy + ii)) + (idx + jj)];
                        }
                    }
                mean = mean / cnt;

                mean_2 = 0;
                cnt = 0;
                for (int ii = 0; ii <= mean; ii++) {
                    mean_2 += ii * hist[ii];
                    cnt += hist[ii];
                }

                if (cnt == 0)
                    mean_2 = mean;
                else
                    mean_2 = mean_2 / cnt;

                result->data[(sz_x * (i)) + (j)] = mean_2;
            }
            idx = 0;
        }
    }

    /**
    * @brief Function for calculate blob filter (positive and negative)
    * @param rad size of blob
    * @param all_mat positive blob filter
    * @param all_mat_neg negative blob filter
    */
    static void gen_blob_neu(int rad, Mat *all_mat, Mat *all_mat_neg) {

        int len = 1 + (4 * rad);
        int c0 = rad * 2;
        float negis = 0;
        float posis = 0;

        *all_mat = Mat::zeros(len, len, CV_32FC1);
        *all_mat_neg = Mat::zeros(len, len, CV_32FC1);

        float *p, *p_neg;
        for (int i = -rad * 2; i <= rad * 2; i++) { //height
            p = all_mat->ptr<float>(c0 + i);

            for (int j = -rad * 2; j <= rad * 2; j++) {

                if (i < -rad || i > rad) { //pos
                    p[c0 + j] = 1;
                    posis++;
                } else { //neg

                    int sz_w = (int) sqrt(float(rad * rad) - float(i * i));

                    if (abs(j) <= sz_w) {
                        p[c0 + j] = -1;
                        negis++;
                    } else {
                        p[c0 + j] = 1;
                        posis++;
                    }
                }
            }
        }

        for (int i = 0; i < len; i++) {
            p = all_mat->ptr<float>(i);
            p_neg = all_mat_neg->ptr<float>(i);

            for (int j = 0; j < len; j++) {

                if (p[j] > 0) {
                    p[j] = (int) 1.0 / posis;
                    p_neg[j] = 0.0;
                } else {
                    p[j] = (int) -1.0 / negis;
                    p_neg[j] = (int) 1.0 / negis;
                }
            }
        }
    }

    /**
    * @brief Function for evaluate ellipse
    * @param ellipse ellipse to evaluate
    * @param pic image
    * @return true if ellipse is good
    */
    static bool is_good_ellipse_evaluation(RotatedRect *ellipse, Mat *pic) {

        if (ellipse->center.x == 0 && ellipse->center.y == 0)
            return false;

        float x0 = ellipse->center.x;
        float y0 = ellipse->center.y;

        int st_x = (int) ceil(x0 - (ellipse->size.width / 4.0));
        int st_y = (int) ceil(y0 - (ellipse->size.height / 4.0));
        int en_x = (int) floor(x0 + (ellipse->size.width / 4.0));
        int en_y = (int) floor(y0 + (ellipse->size.height / 4.0));

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

        st_x = (int) ceil(x0 - (ellipse->size.width * 0.75));
        st_y = (int) ceil(y0 - (ellipse->size.height * 0.75));
        en_x = (int) floor(x0 + (ellipse->size.width * 0.75));
        en_y = (int) floor(y0 + (ellipse->size.height * 0.75));

        int in_st_x = (int) ceil(x0 - (ellipse->size.width / 2));
        int in_st_y = (int) ceil(y0 - (ellipse->size.height / 2));
        int in_en_x = (int) floor(x0 + (ellipse->size.width / 2));
        int in_en_y = (int) floor(y0 + (ellipse->size.height / 2));

        for (int i = st_x; i < en_x; i++)
            for (int j = st_y; j < en_y; j++) {
                if (!(i >= in_st_x && i <= in_en_x && j >= in_st_y && j <= in_en_y))
                    if (i > 0 && i < pic->cols && j > 0 && j < pic->rows) {
                        ext_val += pic->data[(pic->cols * j) + i];
                        val_cnt++;
                    }
            }

        if (val_cnt > 0)
            ext_val = ext_val / val_cnt;
        else
            return false;

        val = ext_val - val;

        if (val > 10)
            return true;
        else
            return false;
    }

    /**
    * @brief Function for find blobs in GPU
    * @param pic original image
    * @return ellipse result
    */
    static RotatedRect blob_findergpu(Mat *pic) {
        Point pos(0, 0);

        int cols = pic->cols;
        int rows = pic->rows;

        int fak_mum = 5;
        int fakk = pic->cols > pic->rows ? (pic->cols / 100) + 1 : (pic->rows / 100) + 1;
        cudaMemset(d_img, 0, (rows / (fak_mum + 1)) * (cols / (fak_mum + 1)) * sizeof(float));
        mumgpu<<<(rows / (fak_mum + 1) * cols / (fak_mum + 1)) / 128 + 1, 128>>>(d_pic, d_img, fak_mum, cols, rows);

        gau2DConv<<<(rows / (fak_mum + 1) * cols / (fak_mum + 1)) / 128 + 1, 128>>>(d_img, d_result, cols / (fak_mum + 1),
                                                                                    rows / (fak_mum + 1), 4 * fakk + 1,
                                                                                    _conv);

        gau2DConv<<<(rows / (fak_mum + 1) * cols / (fak_mum + 1)) / 128 + 1, 128>>>(d_img, d_resultNeg,
                                                                                    cols / (fak_mum + 1),
                                                                                    rows / (fak_mum + 1),
                                                                                    4 * fakk + 1, _convNeg);
        multiply<<<(rows / (fak_mum + 1) * cols / (fak_mum + 1)) / 128 + 1, 128>>>(d_img, d_result, d_resultNeg,
                                                                                cols / (fak_mum + 1),
                                                                                rows / (fak_mum + 1));
        int indexResult = 0;
        //get max value
        cublasIsamax(d_handle, rows / (fak_mum + 1) * cols / (fak_mum + 1), d_img, 1, &indexResult);
        indexResult--;
        pos.x = (indexResult % (cols / (fak_mum + 1))) * (fak_mum + 1) + (fak_mum + 1);
        pos.y = (indexResult / (cols / (fak_mum + 1))) * (fak_mum + 1) + (fak_mum + 1);

        if (pos.y > 0 && pos.y < rows && pos.x > 0 && pos.x < cols) {

            //calc th
            int opti_x = 0;
            int opti_y = 0;

            float mm = 0;
            float cnt = 0;
            for (int i = -(2); i < (2); i++) {
                for (int j = -(2); j < (2); j++) {
                    if (pos.y + i > 0 && pos.y + i < pic->rows && pos.x + j > 0 && pos.x + j < pic->cols) {
                        mm += pic->data[(pic->cols * (pos.y + i)) + (pos.x + j)];
                        cnt++;
                    }
                }
            }

            if (cnt > 0)
                mm = ceil(mm / cnt);

            int th_bot = 0;
            if (pos.y > 0 && pos.y < pic->rows && pos.x > 0 && pos.x < pic->cols)
                th_bot = (int) (pic->data[(pic->cols * (pos.y)) + (pos.x)] +
                                abs(mm - pic->data[(pic->cols * (pos.y)) + (pos.x)]));
            cnt = 0;

            for (int i = -(fak_mum * fak_mum); i < (fak_mum * fak_mum); i++) {
                for (int j = -(fak_mum * fak_mum); j < (fak_mum * fak_mum); j++) {

                    if (pos.y + i > 0 && pos.y + i < pic->rows && pos.x + j > 0 && pos.x + j < pic->cols) {

                        if (pic->data[(pic->cols * (pos.y + i)) + (pos.x + j)] <= th_bot) {
                            opti_x += pos.x + j;
                            opti_y += pos.y + i;
                            cnt++;
                        }
                    }
                }
            }

            if (cnt > 0) {
                opti_x = (int) (opti_x / cnt);
                opti_y = (int) (opti_y / cnt);
            } else {
                opti_x = pos.x;
                opti_y = pos.y;
            }

            pos.x = opti_x;
            pos.y = opti_y;
        }

        RotatedRect ellipse;

        if (pos.y > 0 && pos.y < pic->rows && pos.x > 0 && pos.x < pic->cols) {
            ellipse.center.x = (float) pos.x;
            ellipse.center.y = (float) pos.y;
            ellipse.angle = 0.0;
            ellipse.size.height = (float) ((fak_mum * fak_mum * 2) + 1);
            ellipse.size.width = (float) ((fak_mum * fak_mum * 2) + 1);

            if (!is_good_ellipse_evaluation(&ellipse, pic)) {
                ellipse.center.x = 0;
                ellipse.center.y = 0;
                ellipse.angle = 0;
                ellipse.size.height = 0;
                ellipse.size.width = 0;
            }
        } else {
            ellipse.center.x = 0;
            ellipse.center.y = 0;
            ellipse.angle = 0;
            ellipse.size.height = 0;
            ellipse.size.width = 0;
        }

        return ellipse;
    }

    /**
    * @brief Function for find blobs in CPU
    * @param pic original image
    * @return ellipse result
    */
    static RotatedRect blob_finder(Mat *pic) {

        Point pos(0, 0);
        float abs_max = 0;

        float *p_erg;
        Mat blob_mat, blob_mat_neg;

        int fak_mum = 5;
        int fakk = pic->cols > pic->rows ? (pic->cols / 100) + 1 : (pic->rows / 100) + 1;

        Mat img;
        mum(pic, &img, fak_mum);
        Mat erg = Mat::zeros(img.rows, img.cols, CV_32FC1);

        Mat result, result_neg;

        gen_blob_neu(fakk, &blob_mat, &blob_mat_neg);

        img.convertTo(img, CV_32FC1);
        filter2D(img, result, -1, blob_mat, Point(-1, -1), 0, BORDER_REPLICATE);

        float *p_res, *p_neg_res;
        for (int i = 0; i < result.rows; i++) {
            p_res = result.ptr<float>(i);

            for (int j = 0; j < result.cols; j++) {
                if (p_res[j] < 0)
                    p_res[j] = 0;
            }
        }

        filter2D(img, result_neg, -1, blob_mat_neg, Point(-1, -1), 0, BORDER_REPLICATE);

        for (int i = 0; i < result.rows; i++) {
            p_res = result.ptr<float>(i);
            p_neg_res = result_neg.ptr<float>(i);
            p_erg = erg.ptr<float>(i);

            for (int j = 0; j < result.cols; j++) {
                p_neg_res[j] = (255.0f - p_neg_res[j]);
                p_erg[j] = (p_neg_res[j]) * (p_res[j]);
            }
        }

        for (int i = 0; i < erg.rows; i++) {
            p_erg = erg.ptr<float>(i);

            for (int j = 0; j < erg.cols; j++) {
                if (abs_max < p_erg[j]) {
                    abs_max = p_erg[j];

                    pos.x = (fak_mum + 1) + (j * (fak_mum + 1));
                    pos.y = (fak_mum + 1) + (i * (fak_mum + 1));
                }
            }
        }

        if (pos.y > 0 && pos.y < pic->rows && pos.x > 0 && pos.x < pic->cols) {

            //calc th
            int opti_x = 0;
            int opti_y = 0;

            float mm = 0;
            float cnt = 0;
            for (int i = -(2); i < (2); i++) {
                for (int j = -(2); j < (2); j++) {
                    if (pos.y + i > 0 && pos.y + i < pic->rows && pos.x + j > 0 && pos.x + j < pic->cols) {
                        mm += pic->data[(pic->cols * (pos.y + i)) + (pos.x + j)];
                        cnt++;
                    }
                }
            }

            if (cnt > 0)
                mm = ceil(mm / cnt);

            int th_bot = 0;
            if (pos.y > 0 && pos.y < pic->rows && pos.x > 0 && pos.x < pic->cols)
                th_bot = (int) (pic->data[(pic->cols * (pos.y)) + (pos.x)] +
                                abs(mm - pic->data[(pic->cols * (pos.y)) + (pos.x)]));
            cnt = 0;

            for (int i = -(fak_mum * fak_mum); i < (fak_mum * fak_mum); i++) {
                for (int j = -(fak_mum * fak_mum); j < (fak_mum * fak_mum); j++) {

                    if (pos.y + i > 0 && pos.y + i < pic->rows && pos.x + j > 0 && pos.x + j < pic->cols) {

                        if (pic->data[(pic->cols * (pos.y + i)) + (pos.x + j)] <= th_bot) {
                            opti_x += pos.x + j;
                            opti_y += pos.y + i;
                            cnt++;
                        }
                    }
                }
            }

            if (cnt > 0) {
                opti_x = (int) (opti_x / cnt);
                opti_y = (int) (opti_y / cnt);
            } else {
                opti_x = pos.x;
                opti_y = pos.y;
            }

            pos.x = opti_x;
            pos.y = opti_y;
        }

        RotatedRect ellipse;

        if (pos.y > 0 && pos.y < pic->rows && pos.x > 0 && pos.x < pic->cols) {
            ellipse.center.x = (float) pos.x;
            ellipse.center.y = (float) pos.y;
            ellipse.angle = 0.0;
            ellipse.size.height = (float) ((fak_mum * fak_mum * 2) + 1);
            ellipse.size.width = (float) ((fak_mum * fak_mum * 2) + 1);

            if (!is_good_ellipse_evaluation(&ellipse, pic)) {
                ellipse.center.x = 0;
                ellipse.center.y = 0;
                ellipse.angle = 0;
                ellipse.size.height = 0;
                ellipse.size.width = 0;
            }
        } else {
            ellipse.center.x = 0;
            ellipse.center.y = 0;
            ellipse.angle = 0;
            ellipse.size.height = 0;
            ellipse.size.width = 0;
        }

        return ellipse;
    }

    /**
    * @brief run
    * @param frame
    * @return pupil ellipse
    */
    Pupil run(const Mat &frame) {
        RotatedRect ellipse;
        Point pos(0, 0);

        Mat pic;
        normalize(frame, pic, 0, 255, NORM_MINMAX, CV_8U);

        minArea = frame.cols * frame.rows * minAreaRatio;
        maxArea = frame.cols * frame.rows * maxAreaRatio;

        double border = 0.05; // ER takes care of setting an ROI
        double mean_dist = 3;
        int inner_color_range = 0;

        int start_x = (int) floor(double(pic.cols) * border);
        int start_y = (int) floor(double(pic.rows) * border);

        int end_x = pic.cols - start_x;
        int end_y = pic.rows - start_y;

        Mat picpic = Mat::zeros(end_y - start_y, end_x - start_x, CV_8U);
        Mat magni;

        for (int i = 0; i < picpic.cols; i++) {
            for (int j = 0; j < picpic.rows; j++) {
                picpic.data[(picpic.cols * j) + i] = pic.data[(pic.cols * (start_y + j)) + (start_x + i)];
            }
        }

        Mat detected_edges2 = canny_impl(&picpic, &magni);

        Mat detected_edges = Mat::zeros(pic.rows, pic.cols, CV_8U);
        for (int i = 0; i < detected_edges2.cols; i++)
            for (int j = 0; j < detected_edges2.rows; j++) {
                detected_edges.data[(detected_edges.cols * (start_y + j)) + (start_x + i)] = detected_edges2.data[
                        (detected_edges2.cols * j) + i];
            }

        filter_edges(&detected_edges, start_x, end_x, start_y, end_y);
        ellipse = find_best_edge(&pic, &detected_edges, &magni, start_x, end_x, start_y, end_y, mean_dist,
                                inner_color_range);
    
        if ((ellipse.center.x <= 0 && ellipse.center.y <= 0) || ellipse.center.x >= pic.cols ||
            ellipse.center.y >= pic.rows) {
            ellipse = blob_finder(&pic);
        }

        cv::RotatedRect scaledEllipse(cv::Point2f(ellipse.center.x, ellipse.center.y),
                                    cv::Size2f(ellipse.size.width, ellipse.size.height),
                                    ellipse.angle);

        return Pupil(scaledEllipse);
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
    static void initializeStructures(int cols, int rows, int start_x, int end_x, int start_y, int end_y, int fak_mum) {
        // Global variables
        cudaMalloc((void **) &d_pic, sizeof(unsigned char) * cols * rows);
        cudaMalloc((void **) &d_grayHist, sizeof(unsigned int) * 256);

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

        cudaMalloc((void **) &d_excentricity, sizeof(bool) * cols * rows);
        cudaMalloc((void **) &d_outputImg, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_translation, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_sum_x, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_sum_y, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_total, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_sum_xx, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_sum_yy, sizeof(unsigned int) * cols * rows);
        cudaMalloc((void **) &d_totall, sizeof(unsigned int) * MAX_CURVES);
        cudaMalloc((void **) &d_innerGray, sizeof(unsigned int) * MAX_CURVES);
        cudaMalloc((void **) &d_innerGrayCount, sizeof(unsigned int) * MAX_CURVES);

        cudaMalloc((void **) &d_bb, MAX_CURVES * 5 * sizeof(float));
        cudaMalloc((void **) &d_AA, MAX_CURVES * 3 * MAX_LINE * 5 * sizeof(float));
        cudaMalloc((void **) &d_AAinv, MAX_CURVES * 5 * 5 * sizeof(float));
        cudaMalloc((void **) &d_xx, MAX_CURVES * 5 * sizeof(float));
        cudaMalloc((void **) &aa_bb, MAX_CURVES * sizeof(float *));
        cudaMalloc((void **) &aa_AA, MAX_CURVES * sizeof(float *));
        cudaMalloc((void **) &aa_AAinv, MAX_CURVES * sizeof(float *));
        cudaMalloc((void **) &aa_xx, MAX_CURVES * sizeof(float *));
        cudaMalloc((void **) &d_A_index, MAX_CURVES * sizeof(int));

        //Initialize array of pointers in GPU
        for (int i = 0; i < MAX_CURVES; i++) {
            //aa_bb[i] = d_bb + i * 5;
            float *a = d_bb + i * 5;
            cudaMemcpy(aa_bb + i, &a, sizeof(float *), cudaMemcpyHostToDevice);
            //aa_AA[i] = d_AA + i * 1000 * 5;
            float *b = d_AA + i * 3 * MAX_LINE * 5;
            cudaMemcpy(aa_AA + i, &b, sizeof(float *), cudaMemcpyHostToDevice);
            //aa_AAinv[i] = d_AAinv + i * 5 * 5;
            float *c = d_AAinv + i * 5 * 5;
            cudaMemcpy(aa_AAinv + i, &c, sizeof(float *), cudaMemcpyHostToDevice);
            //aa_xx[i] = d_xx + i * 5;
            float *d = d_xx + i * 5;
            cudaMemcpy(aa_xx + i, &d, sizeof(float *), cudaMemcpyHostToDevice);
        }

        cudaMallocHost((void **) &h_x, 5 * sizeof(float));
        cudaMalloc((void **) &d_info, MAX_CURVES * sizeof(int));

        cudaMalloc((void **) &d_img, rows / (fak_mum + 1) * cols / (fak_mum + 1) * sizeof(float));
        cudaMalloc((void **) &d_result, rows / (fak_mum + 1) * cols / (fak_mum + 1) * sizeof(float));
        cudaMalloc((void **) &d_resultNeg, rows / (fak_mum + 1) * cols / (fak_mum + 1) * sizeof(float));

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
        cudaMemcpyToSymbol(_gauC, gau, sizeof(float) * 16);
        cudaMemcpyToSymbol(_deriv_gauC, deriv_gau, sizeof(float) * 16);

        // Cublas handle
        cublasCreate(&d_handle);

        // Address from symbols
        cudaGetSymbolAddress((void **) &_translationIndex, translationIndex);
        cudaGetSymbolAddress((void **) &_bestEdge, bestLabel);
    }


    /**
    * @brief Free all the memory allocated in the GPU
    */
    static void freeStructures() {
        cudaFree(d_pic);
        cudaFree(d_grayHist);
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
        cudaFree(d_sum_xx);
        cudaFree(d_sum_yy);
        cudaFree(d_totall);
        cudaFree(d_innerGray);
        cudaFree(d_innerGrayCount);
        cudaFree(d_info);


        cudaFree(d_bb);
        cudaFree(d_AA);
        cudaFree(d_AAinv);
        cudaFree(d_xx);
        cudaFree(aa_bb);
        cudaFree(aa_AA);
        cudaFree(aa_AAinv);
        cudaFree(aa_xx);
        cudaFree(d_A_index);

        cudaFree(d_img);
        cudaFree(d_result);
        cudaFree(d_resultNeg);

        cublasDestroy(d_handle);
        cudaDeviceReset();
    }

    /**
    * @brief rungpu
    * @param frame
    * @return
    */
    Pupil rungpu(const Mat &frame, int iteration) {
        RotatedRect ellipse;
        Point pos(0, 0);

        Mat pic;
        normalize(frame, pic, 0, 255, NORM_MINMAX, CV_8U);

        minArea = frame.cols * frame.rows * minAreaRatio;
        maxArea = frame.cols * frame.rows * maxAreaRatio;

        double border = 0.05; // ER takes care of setting an ROI
        double mean_dist = 3;
        int inner_color_range = 0;

        int start_x = (int) floor(double(pic.cols) * border);
        int start_y = (int) floor(double(pic.rows) * border);

        int end_x = pic.cols - start_x;
        int end_y = pic.rows - start_y;

        int rows = frame.rows;
        int cols = frame.cols;

    //    if (iteration == 28) {
        if (iteration == 0) {
            initializeStructures(cols, rows, start_x, end_x, start_y, end_y, 5);
            int fakk = frame.cols > frame.rows ? (frame.cols / 100) + 1 : (frame.rows / 100) + 1;
            Mat blob_mat, blob_mat_neg;
            gen_blob_neu(fakk, &blob_mat, &blob_mat_neg);
            cudaMalloc(&_conv, sizeof(float) * (4 * fakk + 1) * (4 * fakk + 1));
            cudaMalloc(&_convNeg, sizeof(float) * (4 * fakk + 1) * (4 * fakk + 1));
            cudaMemcpy(_conv, blob_mat.data, sizeof(float) * (4 * fakk + 1) * (4 * fakk + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(_convNeg, blob_mat_neg.data, sizeof(float) * (4 * fakk + 1) * (4 * fakk + 1),
                    cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_pic, pic.data, sizeof(unsigned char) * pic.cols * pic.rows, cudaMemcpyHostToDevice);
    
        cudaMemset(d_smallPic, 0, sizeof(float) * (end_x - start_x) * (end_y - start_y));
        removeBorders<<<cols * rows / 256 + 1, 256>>>(d_pic, d_smallPic, cols, rows,
                                                    start_x, end_x, start_y, end_y);

        canny_impl_gpu((end_x - start_x), (end_y - start_y));

        cudaMemset(d_edges, 0, sizeof(unsigned char) * cols * rows);
        addBorders<<<(rows * cols) / 256 + 1, 256>>>(d_exitSmall, d_edges, cols, rows, start_x, end_x,
                                                    start_y, end_y);
        
        filter_edgesgpu(cols, rows, start_x, end_x, start_y, end_y);
        ellipse = find_best_edgegpu(cols, rows, start_x, end_x, start_y, end_y, mean_dist,
                                    inner_color_range);

        if ((ellipse.center.x <= 0 && ellipse.center.y <= 0) || ellipse.center.x >= pic.cols ||
            ellipse.center.y >= pic.rows) {
            ellipse = blob_findergpu(&pic);
        }

        cv::RotatedRect scaledEllipse(cv::Point2f(ellipse.center.x, ellipse.center.y),
                                    cv::Size2f(ellipse.size.width, ellipse.size.height),
                                    ellipse.angle);

        return Pupil(scaledEllipse);
    }
}

extern "C" Pupil ELSEGREEDYI_run(const Mat &frame, int iteration, int gpu) {
    if (gpu) {
        return ElseGreedyI::rungpu(frame, iteration);
    } else {
        return ElseGreedyI::run(frame);
    }
}