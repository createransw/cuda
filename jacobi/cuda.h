#define Max(a, b) ((a) > (b) ? (a) : (b))

#define A(i, j, k) A[((i) * L + (j)) * L + (k)]
#define B(i, j, k) B[((i) * L + (j)) * L + (k)]
#define eps(i, j, k) eps[((i) * L + (j)) * L + (k)]


#define L 100



__global__ void function(const double *A, double *B) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if ((i > 0) && (i < L - 1)) {
        if ((j > 0) && (j < L - 1)) {
            if ((k > 0) && (k < L - 1)) {
                B(i, j, k) = (A(i - 1, j, k) + A(i, j - 1, k) + A(i, j, k - 1) +       A(i, j, k + 1) + A(i, j + 1, k) + A(i + 1, j, k)) / 6.0;
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
                eps(i, j, k) = std::fabs(B(i, j, k) - A(i, j, k));
                A(i, j, k) = B(i, j, k);
            }
        }
    }
}

void launch_function(const double *A, double *B) {
    dim3 blockDim = dim3(16, 8, 4);
    dim3 gridDim = dim3(L / 16 + 1, L / 8 + 1, L / 4 + 1);

    function<<<gridDim, blockDim>>>(A, B);
}


void launch_difference_ab(double *A, const double *B, double *eps) {
    dim3 blockDim = dim3(16, 8, 4);
    dim3 gridDim = dim3(L / 16 + 1, L / 8 + 1, L / 4 + 1);

    difference_ab<<<gridDim, blockDim>>>(A, B, eps);
}
