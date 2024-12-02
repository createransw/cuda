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

#define A(i, j, k) A[((i) * ny + (j)) * nx + (k)]
#define B(i, j, k) B[((i) * ny + (j)) * nx + (k)]
#define eps(i, j, k) eps[((i) * ny + (j)) * nx + (k)]

#define nx 31
#define ny 31
#define nz 31
        

double maxeps = 0.01;
double itmax = 2;

void init(double *a);
double dev(const double *A, const double *B);

__device__ int dim_i[nx / 32 + 1][ny / 32 + 1];
__device__ int dim_j[nx / 32 + 1][nz / 32 + 1];
__device__ int dim_k[ny / 32 + 1][nz / 32 + 1];

__global__ void function(double *A, double *eps, char dim) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (dim == 'i') {
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            while (atomicAdd(&dim_i[gridDim.x][gridDim.y], 0) < i);
        }
        __syncthreads();
        if ((threadIdx.x == 3) && (threadIdx.y == 17))
            printf("%d ", i);
        if ((i > 0) && (i < nx - 1))
            if ((j > 0) && (j < ny - 1))
                if ((k > 0) && (k < nz - 1))
                    A(i, j, k) = (A(i-1, j, k) + A(i+1, j, k)) / 2;
        __syncthreads();
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            __threadfence();
            atomicAdd(&dim_i[gridDim.x][gridDim.y], 1);
        }
    }

    if (dim == 'j') {
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            while (atomicAdd(&dim_j[gridDim.x][gridDim.z], 0) < j);
        }
        __syncthreads();
        if ((i > 0) && (i < nx - 1))
            if ((j > 0) && (j < ny - 1))
                if ((k > 0) && (k < nz - 1))
                    A(i, j, k) = (A(i, j-1, k) + A(i, j+1, k)) / 2; 
        __syncthreads();
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            __threadfence();
            atomicAdd(&dim_j[gridDim.x][gridDim.z], 1);
        }
    }

    if (dim == 'k') {
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            while (atomicAdd(&dim_k[gridDim.y][gridDim.z], 0) < k);
        }
        __syncthreads();
        if ((i > 0) && (i < nx - 1))
            if ((j > 0) && (j < ny - 1))
                if ((k > 0) && (k < nz - 1)) {
                    double tmp = (A(i, j, k-1) + A(i, j, k+1)) / 2;
                    eps(i, j, k) = fabs(A(i, j, k) - tmp);
                    A(i, j, k) = tmp;
                }
        __syncthreads();
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            __threadfence();
            atomicAdd(&dim_k[gridDim.y][gridDim.z], 1);
        }
    }
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


        dim3 blockDim_k = dim3(1, 32, 32);
        dim3 gridDim_k = dim3(nx, ny / 32 + 1, nz / 32 + 1);

        dim3 blockDim_j = dim3(32, 1, 32);
        dim3 gridDim_j = dim3(nx / 32 + 1, ny, nz / 32 + 1);

        dim3 blockDim_i = dim3(32, 32, 1);
        dim3 gridDim_i = dim3(nx / 32 + 1, ny / 32 + 1, nz);

        int *dim_i_ptr; 
        SAFE_CALL(cudaGetSymbolAddress((void**) &dim_i_ptr, dim_i));
        int *dim_j_ptr; 
        SAFE_CALL(cudaGetSymbolAddress((void**) &dim_j_ptr, dim_j));
        int *dim_k_ptr; 
        SAFE_CALL(cudaGetSymbolAddress((void**) &dim_k_ptr, dim_i));


        cudaEvent_t startt, endt;
        SAFE_CALL(cudaEventCreate(&startt));
        SAFE_CALL(cudaEventCreate(&endt));

        double eps = 0;

        SAFE_CALL(cudaEventRecord(startt, 0));
        for (int it = 1; it <= itmax; it++) {
            //std::cerr << "!";
            SAFE_CALL(cudaMemset(dim_i_ptr, 0, sizeof(dim_i)));
            function<<<gridDim_i, blockDim_i>>>(A_device, ptrdiff, 'i');

            //std::cerr << "!";
            SAFE_CALL(cudaMemset(dim_j_ptr, 0, sizeof(dim_j)));
            function<<<gridDim_j, blockDim_j>>>(A_device, ptrdiff, 'j');

            //std::cerr << "!";
            SAFE_CALL(cudaMemset(dim_k_ptr, 0, sizeof(dim_k)));
            function<<<gridDim_k, blockDim_k>>>(A_device, ptrdiff, 'k');

            //std::cerr << "!";
            //std::cerr << it << ' ';

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
