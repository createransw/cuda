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
#define temp_i(i, j, k) temp_i[((i) * 4 + (j)) * 4 + (k)]
#define temp_j(i, j, k) temp_j[((i) * 64 + (j)) * 4 + (k)]
#define temp_k(i, j, k) temp_k[((i) * 4 + (j)) * 64 + (k)]

#define nx 200
#define ny 200
#define nz 200
        

double maxeps = 0.01;
double itmax = 100;

void init(double *a);
double dev(const double *A, const double *B);

__device__ int dim_i[ny / 4 + 1][nz / 4 + 1];
__device__ int dim_j[nx / 4 + 1][nz / 4 + 1];
__device__ int dim_k[nx / 4 + 1][ny / 4 + 1];

__device__ int ord_i[ny / 4 + 1][nz / 4 + 1];
__device__ int ord_j[nx / 4 + 1][nz / 4 + 1];
__device__ int ord_k[nx / 4 + 1][ny / 4 + 1];


__device__ double val_i[ny][nz];
__device__ double val_j[nx][nz];
__device__ double val_k[nx][ny];

__global__ void function_i(double *A) {
    __shared__ int my_block_id;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        my_block_id = atomicAdd(&ord_i[blockIdx.y][blockIdx.z], 1);
    }
    __syncthreads();

    int i = my_block_id * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float temp_i[];

    if (threadIdx.x == 0)
        temp_i(threadIdx.x, threadIdx.y, threadIdx.z) = 0;
    else
        if (i < nx)
            if (j < ny)
                if (k < nz)
                    temp_i(threadIdx.x, threadIdx.y, threadIdx.z) = A(i, j, k) / 4;

    for (int d = 1; d < blockDim.x; d <<= 1) {
        __syncthreads();
        double tmp = (threadIdx.x >= d) ? temp_i(threadIdx.x - d, threadIdx.y, threadIdx.z) : 0;
        __syncthreads();
        temp_i(threadIdx.x, threadIdx.y, threadIdx.z) += (tmp / (1 << d));
    }
    if (i < nx - 1)
        if (j < ny)
            if (k < nz)
                temp_i(threadIdx.x, threadIdx.y, threadIdx.z) += A(i + 1, j, k) / 2;


    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        while (atomicAdd(&dim_i[blockIdx.y][blockIdx.z], 0) < my_block_id);
    }
    __syncthreads();

    if ((i > 0) && (i < nx - 1))
        if ((j > 0) && (j < ny - 1))
            if ((k > 0) && (k < nz - 1))
               A(i, j, k) = temp_i(threadIdx.x, threadIdx.y, threadIdx.z) + val_i[j][k] / (1 << (threadIdx.x + 1));
    __syncthreads();


    if (threadIdx.x == blockDim.x - 1)
        if (j < ny)
            if (k < nz) {
                if (my_block_id == gridDim.x - 1) {
                    val_i[j][k] = A(0, j, k);
                } else {
                    val_i[j][k] = A(i, j, k);
                }
            }

    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        __threadfence();
        atomicAdd(&dim_i[blockIdx.y][blockIdx.z], 1);
        if (my_block_id  == gridDim.x - 1) {
            dim_i[blockIdx.y][blockIdx.z] = 0;
            ord_i[blockIdx.y][blockIdx.z] = 0;
        }
    } 
}


__global__ void function_j(double *A) {
    __shared__ int my_block_id;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        my_block_id = atomicAdd(&ord_j[blockIdx.x][blockIdx.z], 1);
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = my_block_id * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    extern __shared__ float temp_j[];

    if (threadIdx.y == 0)
        temp_j(threadIdx.x, threadIdx.y, threadIdx.z) = 0;
    else
        if (i < nx)
            if (j < ny)
                if (k < nz)
                    temp_j(threadIdx.x, threadIdx.y, threadIdx.z) = A(i, j, k) / 4;

    for (int d = 1; d < blockDim.y; d <<= 1) {
        __syncthreads();
        double tmp = (threadIdx.y >= d) ? temp_j(threadIdx.x, threadIdx.y - d, threadIdx.z) : 0;
        __syncthreads();
        temp_j(threadIdx.x, threadIdx.y, threadIdx.z) += (tmp / (1 << d));
    }
    if (i < nx)
        if (j < ny - 1)
            if (k < nz)
                temp_j(threadIdx.x, threadIdx.y, threadIdx.z) += A(i, j + 1, k) / 2;


    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        while (atomicAdd(&dim_j[blockIdx.x][blockIdx.z], 0) < my_block_id);
    }
    __syncthreads();

    if ((i > 0) && (i < nx - 1))
        if ((j > 0) && (j < ny - 1))
            if ((k > 0) && (k < nz - 1))
               A(i, j, k) = temp_j(threadIdx.x, threadIdx.y, threadIdx.z) + val_j[i][k] / (1 << (threadIdx.y + 1));
    __syncthreads();


    if (threadIdx.y == blockDim.y - 1)
        if (i < nx)
            if (k < nz) {
                if (my_block_id == gridDim.y - 1) {
                    val_j[i][k] = A(i, 0, k);
                } else {
                    val_j[i][k] = A(i, j, k);
                }
            }

    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        __threadfence();
        atomicAdd(&dim_j[blockIdx.x][blockIdx.z], 1);
        if (my_block_id  == gridDim.y - 1) {
            dim_j[blockIdx.x][blockIdx.z] = 0;
            ord_j[blockIdx.x][blockIdx.z] = 0;
        }
    } 
}


__global__ void function_k(double *A, double *eps) {
    __shared__ int my_block_id;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        my_block_id = atomicAdd(&ord_k[blockIdx.x][blockIdx.y], 1);
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = my_block_id * blockDim.z + threadIdx.z + 1;

    extern __shared__ float temp_k[];

    if (threadIdx.z == 0)
        temp_k(threadIdx.x, threadIdx.y, threadIdx.z) = 0;
    else
        if (i < nx)
            if (j < ny)
                if (k < nz)
                    temp_k(threadIdx.x, threadIdx.y, threadIdx.z) = A(i, j, k) / 4;

    for (int d = 1; d < blockDim.z; d <<= 1) {
        __syncthreads();
        double tmp = (threadIdx.z >= d) ? temp_k(threadIdx.x, threadIdx.y, threadIdx.z - d) : 0;
        __syncthreads();
        temp_k(threadIdx.x, threadIdx.y, threadIdx.z) += (tmp / (1 << d));
    }
    if (i < nx)
        if (j < ny)
            if (k < nz - 1)
                temp_k(threadIdx.x, threadIdx.y, threadIdx.z) += A(i, j, k + 1) / 2;


    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        while (atomicAdd(&dim_k[blockIdx.x][blockIdx.y], 0) < my_block_id);
    }
    __syncthreads();

    if ((i > 0) && (i < nx - 1))
        if ((j > 0) && (j < ny - 1))
            if ((k > 0) && (k < nz - 1)) {
                double tmp1 = temp_k(threadIdx.x, threadIdx.y, threadIdx.z) + val_k[i][j] / (1 << (threadIdx.z + 1));
                eps(i, j, k) = fabs(A(i, j, k) - tmp1);
                A(i, j, k) = tmp1;
            }
    __syncthreads();


    if (threadIdx.z == blockDim.z - 1)
        if (i < nx)
            if (j < ny) {
                if (my_block_id == gridDim.z - 1) {
                    val_k[i][j] = A(i, j, 0);
                } else {
                    val_k[i][j] = A(i, j, k);
                }
            }

    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        __threadfence();
        atomicAdd(&dim_k[blockIdx.x][blockIdx.y], 1);
        if (my_block_id  == gridDim.z - 1) {
            dim_k[blockIdx.x][blockIdx.y] = 0;
            ord_k[blockIdx.x][blockIdx.y] = 0;
        }
    } 
}



__global__ void init_i(double *A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i == 0) 
        if (j < ny)
            if (k < nz) 
                val_i[j][k] = 10.0 * j / (ny - 1) + 10.0 * k / (nz - 1);
    dim_i[blockIdx.y][blockIdx.z] = 0;
    ord_i[blockIdx.y][blockIdx.z] = 0;
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

            for (int i = 1; i < nx - 1; i++)
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
                break;
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
        SAFE_CALL(cudaMemcpy(ptrdiff, A_host, size, cudaMemcpyHostToDevice));


        dim3 blockDim_i = dim3(64, 4, 4);
        dim3 blockDim_j = dim3(4, 64, 4);
        dim3 blockDim_k = dim3(4, 4, 64);
        dim3 gridDim_i = dim3(nx / 64 + 1, ny / 4 + 1, nz / 4 + 1);
        dim3 gridDim_j = dim3(nx / 4 + 1, ny / 64 + 1, nz / 4 + 1);
        dim3 gridDim_k = dim3(nx / 4 + 1, ny / 4 + 1, nz / 64 + 1);

        int block_size = 64 * 4 * 4 * sizeof(double);

        init_i<<<gridDim_i, blockDim_i>>>(A_device);
        init_j<<<gridDim_j, blockDim_j>>>(A_device);
        init_k<<<gridDim_k, blockDim_k>>>(A_device);


        cudaEvent_t startt, endt;
        SAFE_CALL(cudaEventCreate(&startt));
        SAFE_CALL(cudaEventCreate(&endt));

        double eps = 0;

        SAFE_CALL(cudaEventRecord(startt, 0));
        for (int it = 1; it <= itmax; it++) {
            function_i<<<gridDim_i, blockDim_i, block_size>>>(A_device);
            function_j<<<gridDim_j, blockDim_j, block_size>>>(A_device);
            function_k<<<gridDim_k, blockDim_k, block_size>>>(A_device, ptrdiff);


            eps = thrust::reduce(diff.begin(), diff.end(), 0.0, thrust::maximum<double>());
            if (eps < maxeps)
                break;
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
        std::cout << "decrease = " << cpu_time / gpu_time * 1000<< std::endl;
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
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
            for (int k = 1; k < nz - 1; k++)
            {
                double tmp = fabs(B(i, j, k) - A(i, j, k));
                delta = Max(tmp, delta);
                if (A(i, j, k) == 0) count++;
            }
    return A == B ? count : delta;
}
