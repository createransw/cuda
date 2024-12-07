/* Jacobi-3 program */

#include <ctime>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <ostream>
#include <thrust/device_vector.h>

#define SAFE_CALL(err) do {                     \
    if (err != 0) {                             \
        std::cerr << "ERROR [" << __FILE__ << "] in line " << __LINE__ << ": " <<       cudaGetErrorString(err) << std::endl;   \
        exit(1);                                \
    }                                           \
} while(0)


#define Max(a, b) ((a) > (b) ? (a) : (b))
#define fabs(a) ((a) > 0 ? (a) : -(a))

#define A(i, j, k) A[((i) * L + (j)) * L + (k)]
#define B(i, j, k) B[((i) * L + (j)) * L + (k)]
#define eps(i, j, k) eps[((i) * L + (j)) * L + (k)]


// #define L 885
#define L 700
#define ITMAX 100

double eps;
double MAXEPS = 0.5;

__global__ void function(const double *A, double *B, double *eps) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if ((i > 0) && (i < L - 1)) {
        if ((j > 0) && (j < L - 1)) {
            if ((k > 0) && (k < L - 1)) {
                B(i, j, k) = (A(i - 1, j, k) + A(i, j - 1, k) + A(i, j, k - 1) + A(i, j, k + 1) + A(i, j + 1, k) + A(i + 1, j, k)) / 6.0;
                eps(i, j, k) = fabs(B(i, j, k) - A(i, j, k));
            }
        }
    }
}

__global__ void difference_ab(double *A, const double *B, double *eps) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if ((i > 0) && (i < L - 1)) {
        if ((j > 0) && (j < L - 1)) {
            if ((k > 0) && (k < L - 1)) {
                eps(i, j, k) = fabs(B(i, j, k) - A(i, j, k));
                A(i, j, k) = B(i, j, k);
            }
        }
    }
}

double dev(const double *A, const double *B) {
    double delta = 0.0;
    for (int i = 1; i < L - 1; i++)
        for (int j = 1; j < L - 1; j++)
            for (int k = 1; k < L - 1; k++)
            {
                double tmp = fabs(B(i, j, k) - A(i, j, k));
                delta = Max(tmp, delta);
            }
    return delta;
}

void set(double *A, double *B) {
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            for (int k = 0; k < L; k++)
            {
                A(i, j, k)= 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B(i, j, k) = 0;
                else
                    B(i, j, k) = 4 + i + j + k;
            }    
}

int main(int an, char **as)
{
    std::cout << "usage:\t\tyacoby -[cg]" << std::endl;

    bool CPU = false;
    bool GPU = true;
    if (an >= 2) {
        GPU = false;
        for (int i = 0; as[1][i] != '\0'; ++i) {
            if (as[1][i] == 'g') GPU = true;
            if (as[1][i] == 'c') CPU = true;
        }
    }

    const long size = L * L * L * sizeof(double);
    
    double *A = (double*)malloc(size);
    double *B = (double*)malloc(size);

    float cpu_time = 0;
    if (CPU) {
        set(A, B);

        /* iteration loop */
        clock_t startt = clock();
        for (int it = 1; it <= ITMAX; it++) {
            eps = 0;
            
            for (int i = 1; i < L - 1; i++)
                for (int j = 1; j < L - 1; j++)
                    for (int k = 1; k < L - 1; k++)
                    {
                        double tmp = fabs(B(i, j, k) - A(i, j, k));
                        eps = Max(tmp, eps);
                        A(i, j, k) = B(i, j, k);
                    }

            for (int i = 1; i < L - 1; i++)
                for (int j = 1; j < L - 1; j++)
                    for (int k = 1; k < L - 1; k++)
                        B(i, j, k) = (A(i - 1, j, k) + A(i, j - 1, k) + A(i, j, k - 1) + A(i, j, k + 1) + A(i, j + 1, k) + A(i + 1, j, k)) / 6.0f;

            if (eps < MAXEPS)
                break;
        }
        clock_t endt = clock();

        cpu_time = float(endt - startt) / CLOCKS_PER_SEC;
    }

    double *A_host = (double*)malloc(size);
    double *B_host = (double*)malloc(size);

    float gpu_time = 0;
    if (GPU) {
        int deviceCount=0;
        cudaGetDeviceCount( &deviceCount ); // число доступных устройств
        if (deviceCount < 1) exit(1);
        cudaSetDevice(0); // Выбрать для работы заданное устройство
        
        set(A_host, B_host);

        double *A_device, *B_device;
        SAFE_CALL(cudaMalloc((void**)&A_device, size));
        SAFE_CALL(cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMalloc((void**)&B_device, size));
        SAFE_CALL(cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice));

        thrust::device_vector<double> diff(L * L * L);
        double *ptrdiff = thrust::raw_pointer_cast(&diff[0]);
        double eps = 0.0;


        dim3 blockDim = dim3(32, 4, 4);
        //int block = blockDim.x * blockDim.y * blockDim.z;
        dim3 gridDim = dim3(L / 32 + 1, L / 4 + 1, L / 4 + 1);

       
        bool flg = true;


        cudaEvent_t startt, endt;
        cudaEventCreate(&startt);
        cudaEventCreate(&endt);

        cudaEventRecord(startt, 0);

        difference_ab<<<gridDim, blockDim>>>(A_device, B_device, ptrdiff);
        flg = not flg;
        eps = thrust::reduce(diff.begin(), diff.end(), 0.0, thrust::maximum<double>());
        /* iteration loop */
        for (int it = 1; it <= ITMAX - 1; it++) {
            if (eps < MAXEPS)
                break;
            if (flg)
                function<<<gridDim, blockDim>>>(A_device, B_device, ptrdiff);
            else 
                function<<<gridDim, blockDim>>>(B_device, A_device, ptrdiff);
            flg = not flg;
            eps = thrust::reduce(diff.begin(), diff.end(), 0.0, thrust::maximum<double>());
        }
        cudaEventRecord(endt, 0);

        cudaEventSynchronize(endt);
        cudaEventElapsedTime(&gpu_time, startt, endt);
        cudaEventDestroy(startt);
        cudaEventDestroy(endt);

        if (flg)
            SAFE_CALL(cudaMemcpy(A_host, A_device, size, cudaMemcpyDeviceToHost));
        else
            SAFE_CALL(cudaMemcpy(A_host, B_device, size, cudaMemcpyDeviceToHost));

        SAFE_CALL(cudaFree(A_device));
        SAFE_CALL(cudaFree(B_device));
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
    }

    free(A);
    free(A_host);
    free(B);
    free(B_host);

    return 0;
}
