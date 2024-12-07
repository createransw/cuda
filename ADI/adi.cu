/* ADI program */

#include <cstdio>
#include <ctime>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>

#define SAFE_CALL(err) do {                     \
    if (err != 0) {                             \
        std::cerr << "ERROR [" << __FILE__ << "] in line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl;         \
        exit(1);                                \
    }                                           \
} while(0)



#define Max(a, b) ((a) > (b) ? (a) : (b))

#define A(i, j, k) A[((i) * ny + (j)) * nz + (k)]
#define A_host(i, j, k) A_host[((i) * ny + (j)) * nz + (k)]
#define B(i, j, k) B[((i) * ny + (j)) * nz + (k)]
#define eps(i, j, k) eps[((i) * ny + (j)) * nz + (k)]
#define temp(i, j, k) temp[((i) * 8 + (j)) * 8 + (k)]

#define nx 50
#define ny 50
#define nz 50
        

double maxeps = 0.01;
double itmax = 1;

void init(double *a);
double dev(const double *A, const double *B);

__device__ int dim_i[ny / 8 + 1][nz / 8 + 1];
__device__ int dim_j[nx / 16 + 1][nz / 8 + 1];
__device__ int dim_k[nx / 16 + 1][ny / 8 + 1];


__device__ double val_i[ny][nz];
__device__ double val_j[nx][nz];
__device__ double val_k[nx][ny];

__global__ void function_i(double *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float temp[];

    if (threadIdx.x == 0)
        temp(threadIdx.x, threadIdx.y, threadIdx.z) = 0;
    else
        if (i < nx)
            if (j < ny)
                if (k < nz)
                    temp(threadIdx.x, threadIdx.y, threadIdx.z) = A(i, j, k) / 4;

    for (int d = 1; d < blockDim.x; d <<= 1) {
        __syncthreads();
        double tmp = (threadIdx.x >= d) ? temp(threadIdx.x - d, threadIdx.y, threadIdx.z) : 0;
        __syncthreads();
        temp(threadIdx.x, threadIdx.y, threadIdx.z) += (tmp / (1 << d));
    }
    if (i < nx - 1)
        if (j < ny)
            if (k < nz)
                temp(threadIdx.x, threadIdx.y, threadIdx.z) += A(i + 1, j, k) / 2;


    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        while (atomicAdd(&dim_i[blockIdx.y][blockIdx.z], 0) < blockIdx.x);
    }
    __syncthreads();

    if ((i > 0) && (i < nx - 1))
        if ((j > 0) && (j < ny - 1))
            if ((k > 0) && (k < nz - 1))
               A(i, j, k) = temp(threadIdx.x, threadIdx.y, threadIdx.z) + val_i[j][k] / (1 << (threadIdx.x + 1));
    __syncthreads();


    if (threadIdx.x == blockDim.x - 1)
        if (j < ny)
            if (k < nz) {
                val_i[j][k] = (blockIdx.x == gridDim.x - 1) ? A(0, j, k) * 2 : A(i, j, k);
            }

    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        __threadfence();
        atomicAdd(&dim_i[blockIdx.y][blockIdx.z], 1);
        if (blockIdx.x  == gridDim.x - 1)
            dim_i[blockIdx.y][blockIdx.z] = 0;
    } 
}


__global__ void function_j(double *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float temp[];

    if ((threadIdx.y == 0) || (j == 1))
        temp(threadIdx.x, threadIdx.y, threadIdx.z) = 0;
    else
        if (i < nx)
            if (j < ny)
                if (k < nz)
                    temp(threadIdx.x, threadIdx.y, threadIdx.z) = A(i, j, k) / 4;

    for (int d = 1; d < blockDim.y; d <<= 1) {
        __syncthreads();
        double tmp = (threadIdx.y >= d) ? temp(threadIdx.x, threadIdx.y - d, threadIdx.z) : 0;
        __syncthreads();
        temp(threadIdx.x, threadIdx.y, threadIdx.z) += (tmp / (1 << d));
    }
    if (i < nx)
        if (j < ny - 1)
            if (k < nz)
                temp(threadIdx.x, threadIdx.y, threadIdx.z) += A(i, j + 1, k) / 2;


    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        while (atomicAdd(&dim_j[blockIdx.x][blockIdx.z], 0) < blockIdx.y);
    }
    __syncthreads();

    if ((i > 0) && (i < nx - 1))
        if ((j > 0) && (j < ny - 1))
            if ((k > 0) && (k < nz - 1))
               A(i, j, k) = temp(threadIdx.x, threadIdx.y, threadIdx.z) + val_j[i][k] / (1 << (threadIdx.y + 1));
    __syncthreads();


    if (threadIdx.y == blockDim.y - 1)
        if (i < nx)
            if (k < nz) {
                val_j[i][k] = (blockIdx.y == gridDim.y - 1) ? A(i, 0, k) * 2 : A(i, j, k);
            }

    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        __threadfence();
        atomicAdd(&dim_j[blockIdx.x][blockIdx.z], 1);
        if (blockIdx.y  == gridDim.y - 1)
            dim_i[blockIdx.x][blockIdx.z] = 0;
    } 
}


__global__ void function_k(double *A, double *eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float temp[];

    if ((threadIdx.z == 0) || (k == 1))
        temp(threadIdx.x, threadIdx.y, threadIdx.z) = 0;
    else
        if (i < nx)
            if (j < ny)
                if (k < nz)
                    temp(threadIdx.x, threadIdx.y, threadIdx.z) = A(i, j, k) / 4;

    for (int d = 1; d < blockDim.z; d <<= 1) {
        __syncthreads();
        double tmp = (threadIdx.z >= d) ? temp(threadIdx.x, threadIdx.y, threadIdx.z - d) : 0;
        __syncthreads();
        temp(threadIdx.x, threadIdx.y, threadIdx.z) += (tmp / (1 << d));
    }
    if (i < nx)
        if (j < ny)
            if (k < nz - 1)
                temp(threadIdx.x, threadIdx.y, threadIdx.z) += A(i, j, k + 1) / 2;


    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        while (atomicAdd(&dim_k[blockIdx.x][blockIdx.y], 0) < blockIdx.z);
    }
    __syncthreads();

    if ((i > 0) && (i < nx - 1))
        if ((j > 0) && (j < ny - 1))
            if ((k > 0) && (k < nz - 1)) {
               double tmp = temp(threadIdx.x, threadIdx.y, threadIdx.z) + val_k[i][j] / (1 << (threadIdx.z + 1));
               eps(i, j, k) = fabs(A(i, j, k) - tmp);
               A(i, j, k) = tmp;
            }
    __syncthreads();


    if (threadIdx.z == blockDim.z - 1)
        if (i < nx)
            if (j < ny) {
                val_k[i][j] = (blockIdx.z == gridDim.z - 1) ? A(i, j, 0) * 2 : A(i, j, k);
            }

    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        __threadfence();
        atomicAdd(&dim_k[blockIdx.x][blockIdx.y], 1);
        if (blockIdx.z  == gridDim.z - 1)
            dim_k[blockIdx.x][blockIdx.y] = 0;
    } 
}


__global__ void init_i(double *A) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (j < ny)
        if (k < nz) 
            val_i[j][k] = 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
}
__global__ void init_j(double *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx)
        if (k < nz) 
            val_j[i][k] = (10.0 * i / (nx - 1) + 10.0 * k / (nz - 1)) * 2;
}
__global__ void init_k(double *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx)
        if (j < ny) 
            val_k[i][j] = (10.0 * i / (nx - 1) + 10.0 * j / (ny - 1)) * 2;
}


int main(int argc, char *argv[])
{
    std::cout << "usage:\t\tadi -[cg]" << std::endl;

    bool CPU = false;
    bool GPU = true;
    if (argc >= 2) {
        GPU = false;
        for (int i = 0; argv[1][i] != '\0'; ++i) {
            if (argv[1][i] == 'g') GPU = true;
            if (argv[1][i] == 'c') CPU = true;
        }
    }

    const long size = nx * ny * nz * sizeof(double);
    double *A = (double*)malloc(size);

    float cpu_time = 0;
    if (CPU) {
        init(A);

        clock_t startt = clock();
        for (int it = 1; it <= itmax; it++) {
            double eps = 0;        
            for (int i = 1; i < nx - 1; i++)
                for (int j = 1; j < ny - 1; j++)
                    for (int k = 1; k < nz - 1; k++)
                        A(i, j, k) = (A(i-1, j, k) + A(i+1, j, k)) / 2;

            /*for (int i = 1; i < nx - 1; i++)
                for (int j = 1; j < ny - 1; j++)
                    for (int k = 1; k < nz - 1; k++)
                        A(i, j, k) = (A(i, j-1, k) + A(i, j+1, k)) / 2; 

            for (int i = 1; i < nx - 1; i++)
                for (int j = 1; j < ny - 1; j++)
                    for (int k = 1; k < nz - 1; k++)
                    {
                        double tmp1 = (A(i, j, k-1) + A(i, j, k+1)) / 2;
                        double tmp2 = fabs(A(i, j, k) - tmp1);
                        eps = Max(eps, tmp2);
                        A(i, j, k) = tmp1;
                    }

            if (eps < maxeps)
                break;*/
        }
        clock_t endt = clock();

        cpu_time = float(endt - startt) / CLOCKS_PER_SEC;
    }

    double *A_host = (double*)malloc(size);
    float gpu_time = 0;
    if (GPU) {
        int deviceCount = 0;
        SAFE_CALL(cudaGetDeviceCount(&deviceCount));
        if (deviceCount < 1) exit(1);
        SAFE_CALL(cudaSetDevice(0));


        init(A_host);

        double *A_device;
        SAFE_CALL(cudaMalloc((void**)&A_device, size));
        SAFE_CALL(cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice));


        thrust::device_vector<double> diff(nx * ny * nz);
        double *ptrdiff = thrust::raw_pointer_cast(&diff[0]);


        dim3 blockDim = dim3(16, 8, 8);
        dim3 gridDim = dim3(nx / 16 + 1, ny / 8 + 1, nz / 8 + 1);

        int block_size = 16 * 8 * 8 * sizeof(double);

        init_i<<<gridDim, blockDim>>>(A_device);
        init_j<<<gridDim, blockDim>>>(A_device);
        init_k<<<gridDim, blockDim>>>(A_device);


        cudaEvent_t startt, endt;
        SAFE_CALL(cudaEventCreate(&startt));
        SAFE_CALL(cudaEventCreate(&endt));

        double eps = 0;

        SAFE_CALL(cudaEventRecord(startt, 0));
        for (int it = 1; it <= itmax; it++) {
            function_i<<<gridDim, blockDim, block_size>>>(A_device);
            //function_j<<<gridDim, blockDim, block_size>>>(A_device);
            /*function_k<<<gridDim, blockDim, block_size>>>(A_device, ptrdiff);


            eps = thrust::reduce(diff.begin(), diff.end(), 0.0, thrust::maximum<double>());
            if (eps < maxeps)
                break;*/
        }
        SAFE_CALL(cudaEventRecord(endt, 0));

        SAFE_CALL(cudaEventSynchronize(endt));
        SAFE_CALL(cudaEventElapsedTime(&gpu_time, startt, endt));
        SAFE_CALL(cudaEventDestroy(startt));
        SAFE_CALL(cudaEventDestroy(endt));

        SAFE_CALL(cudaMemcpy(A_host, A_device, size, cudaMemcpyDeviceToHost));

        SAFE_CALL(cudaFree(A_device));
    }

    if (CPU && GPU) {
        std::cout << "cpu time = " << cpu_time << std::endl;
        std::cout << "gpu time = " << gpu_time * 0.001 <<  std::endl;
        std::cout << "maksimum deviation = " << dev(A, A_host) << std::endl;
    } else if (CPU) {
        std::cout << "cpu time = " << cpu_time << std::endl;
    } else if (GPU) {
        std::cout << "gpu time = " << gpu_time * 0.001 << std::endl;
        std::cout << dev(A_host, A_host) << " zeros in matrix\n";
    }


    free(A);
    free(A_host);

    return 0;
}

void init(double *A)
{
    int i, j, k;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            for (k = 0; k < nz; k++)
                if (k == 0 || k == nz - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1)
                    A(i, j, k) = 10.0 * i / (nx - 1) + 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
                else
                    A(i, j, k) = 0;
}

double dev(const double *A, const double *B) {
    double delta = 0.0;
    int count = 0;
    std::cout << std::endl;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++)
            std::cout << A(i, j, 1) << ' ';
        std::cout << "\t\t\t";
        for (int j = 0; j < ny; j++)
            std::cout << B(i, j, 1) << ' ';
        std::cout << std::endl;
    };
    int I, J, K;
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
            for (int k = 1; k < nz - 1; k++)
            {
                double tmp = fabs(B(i, j, k) - A(i, j, k));
                if (tmp > delta) {
                    I = i;
                    J = j;
                    K = k;
                }

                delta = Max(tmp, delta);

                if (A(i, j, k) == 0) count++;
            }
    std::cout << '(' << I << ',' << J << ',' << K << ')' << std::endl;
    return A == B ? count : delta;
}
